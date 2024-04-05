import os
from typing import List
from src.auto_chat.arguments import EmbeddingModelArguments, PipelineArguments
from src.auto_chat.utils import init_embeddings
from src.auto_chat.prompts import SYSTEM_PROMPT, IR_TOOL_DESCRIPTION
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain.agents import create_openai_functions_agent
from langchain.memory import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class Pipeline:
    """A pipeline which integrates a Vector DB, IR tool, and an LLM for a complete RAG system"""

    def __init__(self, pipeline_args: PipelineArguments, embed_args: EmbeddingModelArguments):
        self.args = pipeline_args
        self.embed_args = embed_args
        self.embed_model = init_embeddings(embed_args=embed_args)
        vector_path = os.path.join(self.args.vector_db_path, embed_args.embedding_model_name)
        self.vector = FAISS.load_local(
            folder_path=vector_path,
            embeddings=self.embed_model,
            allow_dangerous_deserialization=True
        )
        self.args.retriever_search_kwargs.update(dict(k=self.args.num_retrieved_chunks))
        self.retriever = self.vector.as_retriever(search_kwargs=self.args.retriever_search_kwargs)
        self.ir_queries = []
        self.ir_outputs: List[List[str]] = []

        self.ret_tool = create_retriever_tool(
            retriever=self.retriever,
            name="Information_Retrieval",
            description=IR_TOOL_DESCRIPTION,
            document_separator=self.args.document_separator
        )

        self.main_chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        llm = ChatOpenAI(model=self.args.chat_llm)
        agent = create_openai_functions_agent(llm, [self.ret_tool], self.main_chat_prompt)
        agent_executor = AgentExecutor(agent=agent, tools=[self.ret_tool], **self.args.agen_executor_kwarg)

        self.chat_history = ChatMessageHistory()
        self.conversational_agent_executor = RunnableWithMessageHistory(
            agent_executor,
            lambda session_id: self.chat_history,
            input_messages_key="input",
            output_messages_key="output",
            history_messages_key="chat_history",
        )

    def __call__(self, user_input: str):
        """Runs the whole chat pipline

        Args:
            user_input: The user's message in the current round of conversation.
        """

        response = self.conversational_agent_executor.invoke(
            {
                "input": user_input,
            },
            {"configurable": {"session_id": "unused"}},
        )

        query = None
        quotes = []
        if response['intermediate_steps']:
            query = response['intermediate_steps'][0][0].tool_input['query']
            quotes = response['intermediate_steps'][0][1].split(self.args.document_separator)
        self.ir_queries.append(query)
        self.ir_outputs.append(quotes)

        return response


