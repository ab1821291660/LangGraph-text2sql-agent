from typing import Literal

from langchain_community.utilities import SQLDatabase
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain_community.agent_toolkits import SQLDatabaseToolkit##===================================
from sql_graph.my_llm import llm##===================================


db = SQLDatabase.from_uri('sqlite:///chinook.db')
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
tools = toolkit.get_tools()
for tool in tools:##共四个工具##===================================##===================================
    print(f"{tool.name}: {tool.description}\n")
list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
##
get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")
##
run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")



# 示例：创建一个预定义的工具调用
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])


    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")##===================================
    tool_message = list_tables_tool.invoke(tool_call)##===================================
    response = AIMessage(f"Available tables: {tool_message.content}")
    return {"messages": [tool_call_message, tool_message, response]}



# 示例：强制模型创建一个工具调用
def call_get_schema(state: MessagesState):
    # 注意：LangChain强制要求所有模型都接受 `tool_choice="any"`
    # 以及 `tool_choice=<工具名称字符串>` 这两种参数
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")##===================================
    response = llm_with_tools.invoke(state["messages"])##===================================
    return {"messages": [response]}




generate_query_system_prompt = """
你是一个设计用于与SQL数据库交互的智能体。
给定一个输入问题，创建一个语法正确的{dialect}查询来运行，
然后查看查询结果并返回答案。除非用户明确指定他们希望获取的示例数量，
否则始终将查询限制为最多{top_k}个结果。

你可以按相关列对结果进行排序，以返回数据库中最有趣的示例。
永远不要查询特定表的所有列，只询问与问题相关的列。

不要对数据库执行任何DML语句（INSERT、UPDATE、DELETE、DROP等）。
""".format(
    dialect=db.dialect,
    top_k=5,
)
def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # 这里不强制工具调用，允许模型在获得解决方案时自然响应
    llm_with_tools = llm.bind_tools([run_query_tool])##===================================
    response = llm_with_tools.invoke([system_message] + state["messages"])##===================================
    return {"messages": [response]}
check_query_system_prompt = """
你是一个具有高度注意细节能力的SQL专家。
仔细检查{dialect}查询中的常见错误，包括：
- 在NULL值上使用NOT IN
- 应该使用UNION ALL时却使用了UNION
- 在独占范围上使用BETWEEN
- 谓词中的数据类型不匹配
- 正确引用标识符
- 为函数使用正确数量的参数
- 转换为正确的数据类型
- 使用正确的列进行连接

如果存在上述任何错误，请重写查询。如果没有错误，
只需重现原始查询。

在运行此检查后，你将调用适当的工具来执行查询。
""".format(dialect=db.dialect)
def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # 生成一个用于检查的人工用户消息
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="any")##===================================
    response = llm_with_tools.invoke([system_message, user_message])##===================================
    response.id = state["messages"][-1].id
    return {"messages": [response]}




def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"
# 构建状态图
builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")


# 添加边定义状态转移
builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,##===================================
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")
# 编译生成代理
agent = builder.compile()




question = "Which sales agent made the most in sales in 2009?"
# question = "2009年买的最好的曲目是哪个？"
for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values",##===================================
):
    step["messages"][-1].pretty_print()##===================================

