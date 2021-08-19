import logging
import numpy as np
import functools
import operator
import copy
import pandas as pd
from flask import Blueprint, jsonify, request, Flask, render_template
from app.form import TicketForm
from pycarol import Carol, Storage, Query
from pycarol.apps import Apps
from pycarol.filter import Filter, TYPE_FILTER, TERMS_FILTER
from sentence_transformers import SentenceTransformer, util
from webargs import fields, ValidationError
from webargs.flaskparser import parser
from pycarol import Carol, Storage
import torch
import re
import ftfy
from unidecode import unidecode
import ast
import json

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
#_settings = Apps(login).get_settings()

server_bp = Blueprint('main', __name__)

def update_embeddings():
    global df
    
    # Get files from Carol storage
    logger.debug('Loading documents kcs4tickets.')
    df_tmp = storage.load('kcs4tickets', format='pickle', cache=False)
    logger.debug('Done')

    # Update values after all of them are loaded from Carol storage
    df = df_tmp

def transformSentences(m, custom_stopwords):
    # Ensure the parameter type as string
    mproc0 = str(m)
    
    # Set all messages to a standard encoding
    mproc1 = ftfy.fix_encoding(mproc0)
    
    # Replaces accentuation from chars. Ex.: "férias" becomes "ferias" 
    mproc2 = unidecode(mproc1)

    login = Carol()
    _settings = Apps(login).get_settings()
    preproc_mode = _settings.get('preproc_mode')
    
    if preproc_mode == "advanced":
        # Removes special chars from the sentence. Ex.: 
        #  - before: "MP - SIGAEST - MATA330/MATA331 - HELP CTGNOCAD"
        #  - after:  "MP   SIGAEST   MATA330 MATA331   HELP CTGNOCAD"
        mproc3 = re.sub('[^0-9a-zA-Z]', " ", mproc2)
        
        # Sets capital to lower case maintaining full upper case tokens and remove portuguese stop words.
        #  - before: "MP   MEU RH   Horario ou Data registrado errado em solicitacoes do MEU RH"
        #  - after:  "MP MEU RH horario data registrado errado solicitacoes MEU RH"
        mproc4 = " ".join([t.lower() for t in mproc3.split() if t not in custom_stopwords])
        
        return mproc4

    else:
        return mproc2

def findMatches(title, query_tokens, scale):
    title_tokens = title.split()
    
    matches=0
    for qt in query_tokens:
        if qt in title_tokens: 
            matches += 1
            
    return scale * (matches/len(query_tokens))

def keywordSearch(kb, q, nresults=1, threshold=None):
    login = Carol()
    _settings = Apps(login).get_settings()
    keyword_search_fields = _settings.get('keyword_searchfields').split(",")
    keyword_search_fields = [f.lstrip().rstrip() for f in keyword_search_fields]

    title_kb = kb[kb["matched_on"].isin(keyword_search_fields)].copy()
    
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
    
    substr_df["type_of_search"] = "keyword"
    
    return substr_df

def get_similar_questions(df_tmp, query, query_vec, threshold, k, response_columns, id_column="id"):
    dft = df_tmp.copy()

    logger.debug('trying keyword search on title.')
    keywordResults = keywordSearch(kb=dft, q=query, threshold=0.9)

    logger.debug('Converting arrays to tensor.')
    # converting arrays to tensor
    torch_l = [torch.from_numpy(v) for v in dft["sentence_embedding"].values]
    # Converting list of tensors to multidimensional tensor
    articles = torch.stack(torch_l, dim=0)

    logger.debug('Calculating article scores.')
    score = util.pytorch_cos_sim(query_vec, articles)
    dft["score"] = score[0]

    logger.debug('Filtering articles below the threshold.')

    # Whenever a dict is provided we assume the threshold will be defined for particular columns or "all", to designate general
    if isinstance(threshold, dict):

        # If there is a general threshold sets it to start with. 
        # Particular thresholds for each column will be set after the general
        if "all" in threshold:

            logger.debug(f'Using general threshold {threshold["all"]} for all columns without custom threshold.')
            dft["custom_threshold"] = int(threshold["all"])
            del threshold['all']
            
        else:
            dft["custom_threshold"] = None

        # Setting the particular thresholds (per column)
        for c in threshold.keys():

            logger.debug(f'Using {threshold[c]} threshold for {c}.')
            dft.loc[dft["matched_on"] == c, "custom_threshold"] = int(threshold[c])

        dft = dft[dft["score"] >= dft["custom_threshold"] / 100].copy()

    # If the threshold is a single scalar, then it is global
    else:
        dft = dft[dft["score"] >= threshold/100].copy()

    dft["type_of_search"] = "semantic"

    logger.debug('Merging search results.')
    results = pd.concat([keywordResults, dft], ignore_index=True)

    logger.debug('Calculating total number of articles matching the search.')
    total_matches = results[id_column].nunique()

    try:
        results.drop(columns=['sentence_embedding'], inplace=True)
    except:
        pass

    logger.debug('Ranking scores.')
    results.sort_values(by="score", ascending=False, inplace=True)

    logger.debug('Keeping only the highest rank per article.')
    results.drop_duplicates(subset=id_column, keep="first", inplace=True)

    return results[:k], total_matches

df = None
model = None
keywordsearch_flag = False
logger.debug('App started. Please, make sure you load the model and knowledge base before you start.')

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

    # Reading stopwords  to be removed
    logger.info(f'Reading list of custom stopwords.')
    with open('/app/cfg/stopwords.txt') as f:
        custom_stopwords = f.read().splitlines()

    logger.info(f'Translating query \"{query}\" to embedding space.')
    query = transformSentences(query, custom_stopwords)
    query_vec = model.encode([query], convert_to_tensor=True)

    logger.info(f'Calculating similarities.')
    df_res, total_matches = get_similar_questions(df_tmp, query, query_vec, threshold, k, response_columns)

    if len(df_res) < 1:
        logger.warn(f'Unable to find any similar article for the threshold {threshold}.')
        return jsonify({'total_matches': 0, 'topk_results': []})

    logger.info(f'Returning results to request.')
    records_dict = sorted(df_res.to_dict('records'), key=operator.itemgetter('score'), reverse=True)
    
    return jsonify({'total_matches': total_matches, 'topk_results': records_dict})

@server_bp.route('/test_form', methods=["GET", "POST"])
def test_form():
    import requests
    form = TicketForm()

    #if form.validate_on_submit():
    if form.is_submitted():
        result = request.form

        data={'query': result["ticket_subject"], 
            'k': '3', 
            'threshold_custom': {"tags": "80"},
            'response_columns': ['id', 'title', 'html_url', 'score'],
            'filters':[{'filter_field': "module", 
                        "filter_value":  result["ticket_module"]}]
        }

        query_url = "https://sentencesimilarity-kcsfortickets.apps.carol.ai/query"
        r = requests.post(query_url, json=data)

        if r.status_code != 200:
            return render_template("fallback.html")

        results = r.json()["topk_results"]
        if not results:
            return render_template("fallback.html")

        return render_template("showrelatedarticles.html", result=results)

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
