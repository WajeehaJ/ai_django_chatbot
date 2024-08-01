import json

from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
from collections import deque
from .tasks import get_response


class ChatConsumer(WebsocketConsumer):
    chat_history = []
    def receive(self, text_data):
        text_data_json = json.loads(text_data)
        if self.chat_history is not None and len(self.chat_history) >= 10:
            self.chat_history = self.chat_history[-9:]
        #user_history_str = ' '.join(map(str,self.chat_history))
        user_history_str = ""
        get_response.delay(self.channel_name, text_data_json, user_history_str)
        
        async_to_sync(self.channel_layer.send)(
            self.channel_name,
            {
                "type": "chat_message",
                "text": {"msg": text_data_json["text"], "source": "user"},
            },
        )

    def chat_message(self, event):
        text = event["text"]
        if text["source"] == "bot":
            self.chat_history.append("<s>[INST]"+ text['msg'] + "[/INST]</s>")
        else:
            sys_prompt = "<s><<SYS>>You are  a helpful and truthful assistant who provides answers to the user queries and your answers should be truthful,honest and unbiased.<</SYS>>"        
            self.chat_history.append( sys_prompt + "[INST]###Input"+text['msg'] + " [/INST]</s>")

       
        self.send(text_data=json.dumps({"text": text}))