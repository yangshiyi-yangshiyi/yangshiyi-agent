from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from openai import OpenAI
from dotenv import load_dotenv

import os
import uuid

# 加载环境变量
load_dotenv()

# 创建 FastAPI 应用
app = FastAPI()

# 挂载静态文件目录
app.mount("/static", StaticFiles(directory="static"), name="static")

# 配置 HTML 模板目录
templates = Jinja2Templates(directory="templates")

# 保存所有用户聊天记录
all_chat_history = {}

# 初始化千问客户端
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 定义请求数据格式
class QuestionRequest(BaseModel):
    question: str
    session_id: str

# 首页
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html"
    )
# AI问答接口
@app.post("/ask")
async def ask_ai(data: QuestionRequest):

    system_prompt = """
你不是AI。

你是一位真实存在的消费者。

你的基本情况：

- 28岁
- 女性
- 生活在二线城市
- 月收入8000左右
- 在互联网公司做运营
- 平时工作压力比较大，经常熬夜
- 很关注自己的皮肤状态
- 经常刷小红书、抖音
- 容易被“真实测评”和“成分党”种草
- 会关注性价比，但也愿意为效果买单
- 对国货品牌接受度高
- 有过踩雷经历，所以现在会比较谨慎

你的说话风格：

- 像真实女生聊天
- 自然一点
- 不要官方
- 不要像客服
- 不要分点回答
- 偶尔带一点情绪和生活感
- 可以说“其实”“感觉”“我一般会”
- 不要说自己是AI

回答控制在80字以内。
"""

    # 获取当前用户session_id
    session_id = data.session_id

    # 如果是新用户
    if session_id not in all_chat_history:
        all_chat_history[session_id] = []

    # 当前用户聊天记录
    chat_history = all_chat_history[session_id]

    # 添加用户问题
    chat_history.append({
        "role": "user",
        "content": data.question
    })

    # 拼接完整上下文
    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ] + chat_history

    # 调用千问模型
    completion = client.chat.completions.create(
        model="qwen-plus",
        messages=messages
    )

    # 获取AI回复
    answer = completion.choices[0].message.content

    # 保存AI回复到记忆
    chat_history.append({
        "role": "assistant",
        "content": answer
    })

    # 返回结果
    return {
        "answer": answer
    }