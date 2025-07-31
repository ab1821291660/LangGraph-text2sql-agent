from typing import List, Dict
import gradio as gr

from sql_graph.text2sql_graph import make_graph


async def execute_graph(chat_bot: List[Dict]) -> List[Dict]:
    """ 执行工作流的函数"""
    user_input = chat_bot[-1]['content']
    result = ''  # AI助手的最后一条消息
    inputs = {
        "input": user_input
    }
    async with make_graph() as graph:


        async for event in graph.astream({"messages": [{"role": "user", "content": user_input}]},
                                         stream_mode="values"): ##===================================
            messages = event.get('messages')##===================================
            if messages:
                if isinstance(messages, list):
                    message = messages[-1]  # 如果消息是列表，则取最后一个
                if message.__class__.__name__ == 'AIMessage':
                    if message.content:
                        print(result)
                        result = message.content  # 需要在Webui展示的消息
                msg_repr = message.pretty_repr(html=True)##===================================
                print(msg_repr)  # 输出消息的表示形式##===================================
    chat_bot.append({'role': 'assistant', 'content': result})
    return chat_bot


def do_graph(user_input, chat_bot):
    """输入框提交后，执行的函数"""
    if user_input:
        chat_bot.append({'role': 'user', 'content': user_input})
    return '', chat_bot
css = '''
#bgc {background-color: #7FFFD4}
.feedback textarea {font-size: 24px !important}
'''
with gr.Blocks(title='调用MCP服务的TEXT2SQL项目', css=css) as instance:
    gr.Label('调用MCP服务的TEXT2SQL项目', container=False)
    chatbot = gr.Chatbot(type='messages', height=450, label='AI客服')  # 聊天记录组件

    input_textbox = gr.Textbox(label='请输入你的问题📝', value='')  # 输入框组件

                                   ##===================================
    input_textbox.submit(do_graph, [input_textbox, chatbot], [input_textbox, chatbot])\
        .then(execute_graph, chatbot, chatbot)##===================================
if __name__ == '__main__':
    # 启动Gradio的应用
    instance.launch(debug=True)
