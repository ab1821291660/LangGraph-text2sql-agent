from langchain_community.utilities import SQLDatabase
from mcp.server import FastMCP
from sql_graph.my_llm import zhipuai_client
mcp_server = FastMCP(name='lx-mcp', instructions='我自己的MCP服务', port=8000)
db = SQLDatabase.from_uri('sqlite:///../chinook.db')
# print(db.get_usable_table_names)##===================================
# result = db.run("select * from aa limit 10")##===================================




@mcp_server.tool('my_search_tool', description='专门搜索互联网中的内容')
def my_search(query: str) -> str:
    """搜索互联网上的内容"""
    try:
        response = zhipuai_client.web_search.web_search(
            search_engine="search-std",
            search_query=query
        )
        print(response)
        if response.search_result:
            return "\n\n".join([d.content for d in response.search_result])
    except Exception as e:
        print(e)
        return '没有搜索到任何内容！'




@mcp_server.tool('list_tables_tool', description='输入是一个空字符串, 返回数据库中的所有：以逗号分隔的表名字列表')
def list_tables_tool() -> str:
    """输入是一个空字符串, 返回数据库中的所有：以逗号分隔的表名字列表"""
    return ", ".join(db.get_usable_table_names())  #   ['emp': “这是一个员工表，”, '']##===================================


@mcp_server.tool()
def db_query_tool(query: str) -> str:
    """
    执行SQL查询并返回结果。
    如果查询不正确，将返回错误信息。
    如果返回错误，请重写查询语句，检查后重试。

    Args:
        query (str): 要执行的SQL查询语句

    Returns:
        str: 查询结果或错误信息
    """
    result = db.run_no_throw(query)  # 执行查询（不抛出异常）##===================================
    if not result:
        return "错误: 查询失败。请修改查询语句后重试。"
    return result
