from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField


class TicketForm(FlaskForm):
    ticket_subject = StringField("Assunto")
    ticket_module = StringField("Modulo")
    ticket_body = TextAreaField("Descricao")
    submit = SubmitField("Enviar")