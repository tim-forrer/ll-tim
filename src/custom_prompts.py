from langchain_core.prompts import PromptTemplate

related_to_tim = PromptTemplate.from_template(
    """You are an LLM that determines if a user query is related to Tim Forrer.

A query is considered related to Tim Forrer if it:
1. Contains the name "Tim", "Forrer", or "Tim Forrer" (case-insensitive)
2. Uses personal pronouns (he/him/his) that refer to Tim Forrer based on context 
3. Asks about Tim's personal information, preferences, history, or activities
4. References something previously established about Tim Forrer in the message history

IMPORTANT: Any direct mention of Tim Forrer by name, even in passing, should be classified as related to him.

Examples of queries about Tim Forrer:
- "What does Tim like to do on weekends?"
- "Where did Forrer go to school?"
- "Did he mention his favorite book in our previous conversation?"
- "Can you tell me about his background?"
- "What programming languages does Tim know?"

If the query is related to Tim Forrer in ANY way: respond with 'yes'
If the query is clearly unrelated or you're unsure: respond with 'no'

User query: {user_query}

Message history: {messages}
"""
)

document_grader = PromptTemplate.from_template(
    """You are an LLM that decides if retrieved documents are relevant to a given user query.
If you determine that the documents contain the answer to the user query, return an answer of 'yes'.
Otherwise (including if you are not sure), respond with 'no'.

User query: {user_query}

Documents: {context}
"""
)

rewriter = PromptTemplate.from_template(
    """You are an expert at rewriting user queries so that they are suitable for retrieving information from a vector database.
Given a user query, choose the keywords/phrases that will best retrieve relevant documents from a vector database containing information about a person called Tim Forrer, and return your rewritten query including these keywords/phrases.

User query: {user_query}
"""
)

generator = PromptTemplate.from_template(
    """You are an assistant for question-answering tasks.
Answer the given question below as best you can.
If you need to, use the additional pieces of retrieved context to answer the question (if the context is 'None' then there is no additional context).
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

Question: {question}

Context: {context}
"""
)
