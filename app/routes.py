import logging
import numpy as np
import pandas as pd
import operator
from flask import Blueprint, jsonify, request, Flask, render_template
from app.form import TicketForm
from pycarol import Carol, Storage
from pycarol.apps import Apps
from sentence_transformers import SentenceTransformer, util
from webargs import fields, ValidationError
from webargs.flaskparser import parser
from unidecode import unidecode
import torch
import re
import ftfy

# Logger
logger = logging.getLogger(__name__)
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(levelname)s: %(message)s")
console.setFormatter(formatter)
logger.addHandler(console)

login = Carol()
storage = Storage(login)

server_bp = Blueprint('main', __name__)

def update_embeddings():
    global df

    login = Carol()
    _settings = Apps(login).get_settings()
    kb_file = _settings.get('knowledgebase_file')
    if kb_file in [None, ""]: 
        logger.warn(f'Setting \"knowledgebase_file\" not filled. Using \"df\" as default.')
        kb_file = "df"
    
    # Get files from Carol storage
    logger.debug(f'Loading documents {kb_file}.')
    df_tmp = storage.load(kb_file, format='pickle', cache=False)
    logger.debug('Done')

    # Update values after all of them are loaded from Carol storage
    df = df_tmp

def transformSentences(m, custom_stopwords):
    login = Carol()
    _settings = Apps(login).get_settings()
    preproc_mode = _settings.get('preproc_mode')

    # Ensure the parameter type as string
    mproc0 = str(m)
    
    # Set all messages to a standard encoding
    mproc1 = ftfy.fix_encoding(mproc0)
    
    # Replaces accentuation from chars. Ex.: "férias" becomes "ferias" 
    mproc2 = unidecode(mproc1)
    
    if preproc_mode.lower() in ["advanced", "stopwords"]:
        # Removes special chars from the sentence. Ex.: 
        #  - before: "MP - SIGAEST - MATA330/MATA331 - HELP CTGNOCAD"
        #  - after:  "MP   SIGAEST   MATA330 MATA331   HELP CTGNOCAD"
        mproc3 = re.sub('[^0-9a-zA-Z]', " ", mproc2)

        if preproc_mode.lower() in ["stopwords"]:
        
            # Sets capital to lower case maintaining full upper case tokens and remove portuguese stop words.
            #  - before: "MP   MEU RH   Horario ou Data registrado errado em solicitacoes do MEU RH"
            #  - after:  "MP MEU RH horario data registrado errado solicitacoes MEU RH"
            mproc4 = " ".join([t.lower() for t in mproc3.split() if t not in custom_stopwords])
        
            return mproc4
        else:
            return mproc3

    else:
        return mproc2


def findMatches(title, query_tokens, scale):
    title_tokens = title.split()
    
    matches=0
    for qt in query_tokens:
        if qt in title_tokens: 
            matches += 1

    return scale * (matches/len(query_tokens))

def keywordSearch(kb, q, nresults=1, threshold=None, fields=["title", "question"]):
    title_kb = kb[kb["sentence_source"].isin(fields)].copy()
    
    # Checks if any article contains the exact query as substring
    substr_df = title_kb[title_kb["sentence"].str.contains(q)].copy()
    
    if len(substr_df) >= 1:
        substr_df["score"] = 1.0
    
    # If the query is not a substring of the title...
    else:
        # ... try moken matches individualy
        query_tokens = q.split()
        title_kb["score"] = title_kb["sentence"].apply(lambda s: findMatches(s, query_tokens, scale=0.9))
        title_kb.sort_values(by="score", ascending=False, inplace=True)
        substr_df = title_kb.copy()
    
    if threshold:
        substr_df = substr_df[substr_df["score"] >= threshold].copy()
    else:
        substr_df =  substr_df.head(nresults)
    
    ## KCS SPECIFIC ##
    # Discount score by its length difference to query
    substr_df["s_length"] = [ len(s) for s in substr_df["sentence"].values ]
    query_length = len(q)
    max_length = substr_df["s_length"].max()
    substr_df["score"] = substr_df["s_length"].apply(lambda x: max(0.80, 1 - (x - query_length)/max_length))
    substr_df.drop(columns=["s_length"], inplace=True)
    ## KCS SPECIFIC ##

    substr_df["type_of_search"] = "keyword"
    
    return substr_df

def matching_tokens(search_tokens, sentence):
    sentence_tokens = sentence.split()
    intersec = list(set(search_tokens) & set(sentence_tokens))
    return len(intersec)/len(search_tokens) * len(intersec)/len(sentence_tokens)

def reverseKeywordSearch(kb, q, fields=["title", "question"], threshold=0.01):
    article_sentences = kb[kb["sentence_source"].isin(fields)].copy()
    query_tokens = q.split()

    # Checks if any article contains the exact query as substring
    substr_df = article_sentences[article_sentences["sentence"].apply(lambda x: True if str(x) in q and len(x) > 10 else False)].copy()

    # If sentence is a substring of the query and it has at least 10 chars
    if len(substr_df) >= 1:
        # HARD CODED above current threshold
        substr_df["score"] = 0.75
        substr_df["type_of_search"] = "rkeyword"
        return substr_df
    
    # Weighting matches over the lengh of search and sentence
    article_sentences["score"] = article_sentences["sentence"].apply(lambda x: matching_tokens(query_tokens, x))
    article_sentences = article_sentences[article_sentences["score"] > (threshold/100)]
    article_sentences["type_of_search"] = "rkeyword"

    return article_sentences

def findCodes(txt):
    # The first pattern captures de definition + numbers, such as "erro d2233".
    pattern1 = re.compile(r'(?:rotina|rejeicao|registro|error|erro|evento)\s[a-z]*\s*[0-9.]{3,}')
    
    # The second pattern, an special case observed on rejections, captures de numbers + definition, such as "934 rejeicao".
    pattern2 = re.compile(r'[0-9]{3,}[\s]+(?:rejeicao)')
    
    codes = pattern1.findall(txt)

    # Handling spaces within error codes
    revised_codes = []
    for c in codes:
        tokens = c.split()
        if len(tokens) > 2:
            # if code comes as ""erro mata 930" it is transformed to "erro mata930"
            revised_codes.append(tokens[0] + " " + "".join(tokens[1:]))
        else:
            revised_codes.append(c)
    
    codesr = pattern2.findall(txt)
    codes = revised_codes + [" ".join(reversed(c.split())) for c in codesr]
    
    return codes

def filterContainingCodes(search_codes, sentence):
    sentence_codes = findCodes(sentence)

    # If the sentence contains a code, present it only if at least one of the 
    # codes match to the codes on search
    if sentence_codes:
        intersec = list(set(sentence_codes) & set(search_codes))
        if intersec:
            return True
        else:
            return False

    # If sentence doesn't have code delegate the decision to semantic
    else:
        return True

## TDN SPECIFIC ##
# Intercept well known failure cases from the model and
# filter out bad answers before sending them to the user
def isRelated(query_ntokens, row, limit):
    try:
        sentence = row["sentence"]
        score = row["score"]
        sentence_tokens = str(sentence).split()

        # if strings have the same lenght there's a higher probability 
        # that sentences are um related
        if (query_ntokens == len(sentence_tokens)) and (score < limit):
            return False
    except:
        pass

    return True

def get_similar_questions(model, sentence_embeddings_df, query, threshold, k, response_columns=None, id_column="id", validation=False):
    global keywordsearch_flag
    global custom_stopwords

    logger.info(f'Translating query \"{query}\" to embedding space.')
    query = transformSentences(query, custom_stopwords)
    query_expanded = [query]

    # KEYWORD SEARCH
    # =================================================
    keyword_columns = ["title", 
                        "title-sinonimos",
                        "question",
                        "question-sinonimos" ,
                        "tags", 
                        "tags-sinonimos", 
                        "autotag"]

    if keywordsearch_flag:
        kcs_articles = sentence_embeddings_df[sentence_embeddings_df["database"] == "KCS"]
        kcskeywordResults = keywordSearch(kb=kcs_articles, q=query, threshold=0.9, fields=keyword_columns)
        kcsrkeywordResults = reverseKeywordSearch(kb=kcs_articles, q=query, fields=keyword_columns, threshold=threshold["all"])
        logger.info(f'{kcskeywordResults.shape[0]} articles retrieved from keyword search on KCS database.')
        logger.info(f'{kcsrkeywordResults.shape[0]} articles retrieved from reverse keyword search on KCS database.')

        tdn_articles = sentence_embeddings_df[sentence_embeddings_df["database"] == "TDN"]
        tdnkeywordResults = keywordSearch(kb=tdn_articles, q=query, threshold=0.9, fields=["title"])
        logger.info(f'{tdnkeywordResults.shape[0]} articles retrieved from keyword search on TDN database.')

        keywordResults = pd.concat([kcskeywordResults, kcsrkeywordResults, tdnkeywordResults])

    else: 
        keywordResults = pd.DataFrame(columns=sentence_embeddings_df.columns)


    # SEMANTIC SEARCH
    # =================================================
    logger.debug('Trying semantic search.')
    logger.debug('Converting arrays to tensor.')
    semanticResults = sentence_embeddings_df.copy()
    torch_l = [torch.from_numpy(v) for v in semanticResults["sentence_embedding"].values]
    articles = torch.stack(torch_l, dim=0)
    logger.debug('Calculating article scores.')
    query_vec = model.encode(query_expanded, convert_to_tensor=True)
    score = util.pytorch_cos_sim(query_vec, articles)

    # from all the query variations, consider only the highest score
    semanticResults['score'] = score.max(dim=0)[0].tolist()
    semanticResults["type_of_search"] = "semantic"
    max_sem_score = semanticResults["score"].max()

    logger.info(f'Maximum semantic score found: {max_sem_score}.')


    # MERGING RESULTS
    # =================================================
    ## TDN SPECIFIC ##
    logger.info('Applying business logic to filter out bad answers fot TDN.')
    tdn_articles = semanticResults[semanticResults["database"] == "TDN"].copy()
    kcs_articles = semanticResults[semanticResults["database"] == "KCS"].copy()

    tdn_previous = tdn_articles[id_column].nunique()
    query_ntokens = len(str(query).split())
    tdn_articles["IsRelated"] = tdn_articles.apply(lambda x: isRelated(query_ntokens=query_ntokens, row=x, limit=0.8), axis=1)
    tdn_articles = tdn_articles[tdn_articles["IsRelated"] != False].copy()

    tdn_after = tdn_articles[id_column].nunique()
    tdn_articles.drop(columns="IsRelated", inplace=True)
    semanticResults = pd.concat([kcs_articles, tdn_articles])
    logger.info(f'{(tdn_previous-tdn_after)} TDN articles discarded.')
    ## TDN SPECIFIC ##

    if (len(keywordResults) > 3) and (keywordResults["score"].mean() == 1.0):
        # if there is any exact match on keywords, ignore semantic search
        logger.info('Exact matches found for query code. Using only keyword search results.')
        results = keywordResults.copy()
    else:
        query_codes = findCodes(query)
        if query_codes:
            logger.info('Query contains code. Using only semantic results containing the same code.')
            semanticResults = semanticResults[semanticResults["sentence"].apply(lambda x: filterContainingCodes(search_codes=query_codes, sentence=x))]

        logger.info('Combining keyword and semantic search results.')
        # if keyword search didn't succeed returns a mix between semantic and keyword search results
        results = pd.concat([keywordResults, semanticResults], ignore_index=True)


    logger.info(f'Evaluating scores against threshold for {results.shape[0]} matching sentences.')

    # Whenever a dict is provided we assume the threshold will be defined for particular columns or "all", to designate general
    if isinstance(threshold, dict):

        # If there is a general threshold sets it to start with. 
        # Particular thresholds for each column will be set after the general
        if "all" in threshold:

            logger.info(f'Using general threshold {threshold["all"]} for all columns without custom threshold.')
            results["custom_threshold"] = int(threshold["all"])
            
        else:
            results["custom_threshold"] = None

        # Setting the particular thresholds (per column)
        for c in threshold.keys():
            if c == "all": continue

            logger.info(f'Using {threshold[c]} threshold for {c}.')
            results.loc[results["sentence_source"] == c, "custom_threshold"] = int(threshold[c])

        results = results[results["score"] >= results["custom_threshold"] / 100].copy()

    # If the threshold is a single scalar, then it is global
    else:
        logger.info(f'Using general {threshold} threshold for all columns.')
        results = results[results["score"] >= (threshold/100)].copy()

    if (len(results) < 1):
        logger.warn('No matching article has been found.')
        return results, 0

    try:
        results.drop(columns=['sentence_embedding'], inplace=True)
    except:
        pass

    logger.debug('Ranking scores.')
    results.sort_values(by="score", ascending=False, inplace=True)

    logger.debug('Keeping only the highest rank per article.')
    if not validation: results.drop_duplicates(subset=['id'], keep="first", inplace=True)

    logger.debug('Calculating total number of articles matching the search.')
    total_matches = results[id_column].nunique()

    return results.head(k), total_matches

# Initialize variables
df = None
model = None
keywordsearch_flag = True

# Reading stopwords  to be removed
logger.info(f'Reading list of custom stopwords.')
with open('/app/cfg/stopwords.txt') as f:
    custom_stopwords = f.read().splitlines()

logger.info('App started. Please, make sure you load the model and knowledge base before you start.')

@server_bp.route('/', methods=['GET'])
def ping():
    return jsonify('App is running. Send a request to /query for document searching or to /update_embeddings to update the document embeddings.')

# Alows to enable/ disable keyword search on run time
@server_bp.route('/switch_keywordsearch', methods=['GET'])
def switch_keywordsearch():
    global keywordsearch_flag

    keywordsearch_flag = not keywordsearch_flag
    return jsonify(f'Keyword search switch set to: {keywordsearch_flag}.')

@server_bp.route('/load_model', methods=['GET'])
def load_model():
    global model

    login = Carol()
    _settings = Apps(login).get_settings()
    model_storage_file = _settings.get('model_storage_file')
    model_sentencetransformers = _settings.get('model_sentencetransformers')

    try:
        gpu = torch.cuda.is_available()
    except Exception as e:
        gpu = False

    if model_storage_file != "":
        name = model_storage_file
        logger.debug(f'Loading model {name}. Using GPU: {gpu}.')
        storage = Storage(login)
        model = storage.load(model_storage_file, format='pickle', cache=False)

        if gpu: 
            model.to(torch.device('cuda'))
            model._target_device = torch.device('cuda')
        else: 
            model.to(torch.device('cpu'))
            model._target_device = torch.device('cpu')
        
    else:
        name = model_sentencetransformers
        logger.debug(f'Loading model {name}. Using GPU: {gpu}.')
        model = SentenceTransformer(model_sentencetransformers)

    return jsonify(f'Model {name} loaded.')

@server_bp.route('/update_embeddings', methods=['GET'])
def update_embeddings_route():
    logger.debug('Updating embeddings.')
    update_embeddings()
    return jsonify('Embeddings are updated.')

# Route to be used for validation purposes. The user can send
# a query and expected results, the response will be the top
# matches on the provided article(s) and their respective sco
# res.
@server_bp.route('/validate', methods=['POST'])
def validate():

    query_arg = {
        "query": fields.Str(required=True, 
            description='Query to be searched in the documents.'),
        "expected_ids": fields.List(fields.Str(), required=True, description='List of expected articles to compare to query.')
    }

    args = parser.parse(query_arg, request)
    query = args['query']
    expected_ids = args['expected_ids']
    expected_ids = [int(i) for i in expected_ids]

    df_tmp = df[df["id"].isin(expected_ids)].copy()
    if len(df_tmp) > 0:

        logger.info(f'Validating query {query} against the following articles: {expected_ids}')
        results_df, total_matches = get_similar_questions(model,
                                                        sentence_embeddings_df=df_tmp, 
                                                        query=query, 
                                                        threshold={"all": 0.0}, 
                                                        k=500, 
                                                        response_columns=None, 
                                                        id_column="id",
                                                        validation=True)

        records_dict_tmp = results_df.to_dict('records')
        records_dict = sorted(records_dict_tmp, key=operator.itemgetter('score'), reverse=True)
    else:
        records_dict, total_matches = ([], 0)

    return jsonify({'total_matches': total_matches, 'topk_results': records_dict})

@server_bp.route('/query', methods=['POST'])
def query():
    
    if model is None:
        logger.debug(f'It looks like the NLP model has not been loaded yet. This operation usually takes up to 1 minute.')
        load_model()

    if df is None:
        logger.debug(f'It looks like the knowledge base has not been loaded yet. This operation usually takes up to 1 minute.')
        update_embeddings()

    # Try to parse threshold as a dict on the first attempt
    query_arg = {
        "query": fields.Str(required=True, description='Query to be searched in the documents.'),
        "k": fields.Int(required=False, missing=5, description='Number of similar documents to be return. Default: 5.'),
        "filters": fields.List(fields.Dict(keys=fields.Str(), values=fields.Raw(), required=False), required=False, missing=None, validate=validate_filter, description='List of dictionaries \
            in which the filter_field means the name of the field and the filter_value the value used for filtering the documents.'),
        "response_columns": fields.List(fields.Str(), required=False, missing=None, validate=validate_response_columns, description='List of columns \
            from the documents base that should be returned in the response.'),
        "threshold": fields.Int(required=False, missing=55, description='Documents with scores below this threshold are not considered similar to the query. Default: 55.'),
        "threshold_custom": fields.Dict(keys=fields.Str(), values=fields.Raw(), required=False, missing=None, validate=validate_threshold_custom, description='Dictionary in which the key is the source from the document in which the sentences has been taking and the values is the the threshold to be considered for that group of sentences.')
        }

    logger.debug('Parsing parameters.')

    args = parser.parse(query_arg, request)
    threshold = args['threshold']
    threshold_custom = args.get('threshold_custom')
    query = args['query']
    k = args['k']
    filters = args['filters']
    response_columns = args['response_columns']

    logger.debug('Consolidating thresholds.')
    # If there's a custom threshold defined it overcomes the general threshold
    if threshold_custom:
        # If there's no "all" key on custom threshold, sets it to the general threshold provided (or default)
        if "all" not in threshold_custom:
            threshold_custom["all"] = threshold
        threshold = threshold_custom
    else:
        threshold = {"all": threshold}

    logger.debug(f'Processing query.')
    df_tmp = df.copy()

    logger.debug(f'Running filter on {len(df_tmp)} articles.')

    # Expceting filters to be passed as a list of dicts as in the example below:
    #     [{'filter_field': 'modulo', 'filter_value': "ARQUIVO SIGAFIS"}] 
    for filter in filters:
        if df_tmp.empty:
            break
        
        filter_field, filter_value = (filter.get('filter_field'), filter.get('filter_value'))

        logger.debug(f'Applying filter \"{filter_field}\" == \"{filter_value}\".')

        filter_field_type = df_tmp.iloc[0][filter_field]

        if isinstance(filter_field_type, list) and isinstance(filter_value, list) and filter_value:
            logger.debug(f'Processing list to list filter.')

            tmp_dfs = []
            for value in filter_value:
                value = value.lower()
                tmp_df = df_tmp[([any(value == v.lower() for v in values) for values in df_tmp[filter_field]])]
                if not tmp_df.empty:
                    tmp_dfs.append(tmp_df)
            if tmp_dfs:
                final_df = pd.concat(tmp_dfs)
                df_tmp = final_df

        elif isinstance(filter_field_type, str) and isinstance(filter_value, list):
            logger.debug(f'Processing string to list filter.')
            df_tmp = df_tmp[df_tmp[filter_field].isin(filter_value)]

        else:
            logger.debug(f'Processing string to string filter.')
            df_tmp = df_tmp[df_tmp[filter_field] == filter_value]

    if not df_tmp.empty:
        logger.debug(f'Total records satisfying the filter: {len(df_tmp)}.')

    else:
        logger.warn(f'No results returned from filter.')
        return jsonify({'total_matches': 0, 'topk_results': []})

    logger.info(f'Calculating similarities.')
    df_res, total_matches = get_similar_questions(model, df_tmp, query, threshold, k, response_columns)

    if len(df_res) < 1:
        logger.warn(f'Unable to find any similar article for the threshold {threshold}.')
        return jsonify({'total_matches': 0, 'topk_results': []})

    logger.info(f'Returning results to request.')
    records_dict = sorted(df_res.to_dict('records'), key=operator.itemgetter('score'), reverse=True)
    
    return jsonify({'total_matches': total_matches, 'topk_results': records_dict})

def formatResultsHTML(results_in):
    results_out = []

    for a in results_in:
        tmp = {}
        tmp["score"] = round(float(a["score"]), 2)

        title = a["title"]
        url = a["html_url"]
        tmp["article"] = f"<a href=\"{url}\">{title}</a>"
        
        tmp["sanitized_solution"] = a["sanitized_solution"]
        results_out.append(tmp)

    return results_out

@server_bp.route('/test_form', methods=["GET", "POST"])
def test_form():
    import requests
    form = TicketForm()

    #if form.validate_on_submit():
    if form.is_submitted():
        result = request.form

        subject = result["ticket_subject"]
        module = result["ticket_module"]
        logger.info(f'Evaluating results for {subject}, on module {module}.')
        data={'query': subject, 
            'k': '3', 
            'threshold': "40",
            'response_columns': ['id', 'title', 'html_url', 'sanitized_solution'],
            'filters':[{'filter_field': "module", 
                        "filter_value": module}]
        }

        logger.info(f'Sending request...')
        query_url = "https://protheusassistant-carolinaticketsapi.apps.carol.ai/query"
        r = requests.post(query_url, json=data)

        if r.status_code != 200:
            logger.warn(f'Bad response from the server: Status {r.status_code}')
            logger.warn(f'Server response: {r.content}')
            return render_template("fallback.html")

        results = r.json()["topk_results"]
        if not results:
            logger.info(f'Unable to find results.')
            return render_template("fallback.html")

        nres = len(results)
        logger.info(f'Presenting {nres} results.')
        #logger.debug(f'Results: {results}')

        #results2show = formatResultsHTML(results)
        return render_template("showrelatedarticles.html", result=results)

    logger.info(f'Presenting test form.')
    return render_template("ticketform.html", form=form)

@server_bp.errorhandler(422)
@server_bp.errorhandler(400)
def handle_error(err):
    headers = err.data.get("headers", None)
    messages = err.data.get("messages", ["Invalid request."])
    messages = messages.get('json', messages)
    if headers:
        return jsonify(messages), err.code, headers
    else:
        return jsonify(messages), err.code

def validate_filter(val):
    logger.debug(str(df.columns))
    filter_columns = []
    filters = list(val)
    for filter in filters:
        filter_field = filter.get('filter_field')
        if filter_field:
            filter_columns.append(filter_field)
        else:
            raise ValidationError("The key 'filter_field' must be filled when you are using filters.") 
    if filters and any(c not in df.columns for c in filter_columns):
        raise ValidationError("One or more columns that you are trying to filter does not exist in the documents base.")


def validate_response_columns(val):
    response_columns = list(val)
    if response_columns and any(c not in df.columns for c in response_columns):
        raise ValidationError("One or more columns that you are trying to return does not exist in the documents base.")

def validate_threshold_custom(val):
    logger.debug('Validating custom threshold')
    if 'matched_on' not in df.columns:
        raise ValidationError("The matched_on column does not exist in the documents base so it will not be possible to filter the custom threshold.")
    sentence_source_values = list(df['matched_on'].unique()) + ["all"]
    sentence_source_filter_values = val.keys()
    if sentence_source_values and any(s not in sentence_source_values for s in sentence_source_filter_values):
        raise ValidationError("One or more values that you are trying to filter does not exist in the matched_on column.")
