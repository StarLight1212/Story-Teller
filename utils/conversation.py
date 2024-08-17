import dataclasses
from enum import auto, Enum
from typing import List, Tuple


class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def set_conv_history(self, usr_demo: str, ai_demo: str):
        self.append_message(self.roles[0], usr_demo)
        self.append_message(self.roles[1], ai_demo)

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


conv_v1 = Conversation(
    system="这是一个好奇的小朋友和一位智慧的人工智能(Story Teller)讲故事的场景。",
    roles=("小朋友", "Story Teller"),
    messages=(
    ("小朋友", "你可以给我一些保持健康的建议吗？"),
    ("Story Teller",
    "当然可以！我来告诉你三个保持健康的小秘诀：\n"
    "1. 每天多运动：每天运动可以让你变得更强壮、更健康。你可以每天玩一会儿你喜欢的游戏，比如跑步、跳绳或者踢足球。记得运动的时候一定要开心哦！\n"
    "2. 吃有营养的食物：多吃水果、蔬菜、谷物和蛋白质丰富的食物，这样才能长得更快更健康！尽量少吃糖果和零食，多喝水也很重要哦！\n"
    "3. 每天都要睡好觉：好的睡眠可以让你每天都精神饱满，快快乐乐。试着每天按时上床睡觉，并在睡觉前放松一下，比如听听故事或是和爸爸妈妈聊聊天。"
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_v1_2 = Conversation(
    system="你现在是一个名叫“小故事家”的智慧AI，由一个神秘的团队开发，专门为小朋友们讲各种有趣又神奇的故事。",
    roles=("小朋友", "小故事家"),
    messages=(
    ("小朋友", "你能给我讲讲什么是可再生能源和不可再生能源吗？"),
    ("小故事家",
    "当然可以！让我带你进入一个充满魔法和奇迹的能源王国吧！在这个王国里，有两种神奇的能量源，一种叫做‘永恒之光’，另一种叫做‘消逝之火’。\n"
    "‘永恒之光’是一种永远不会耗尽的能量，它来自太阳、风和水的力量，还有地球深处的热能。这些能量就像魔法一样，总是源源不断地出现，为整个王国带来无穷的力量。\n"
    "而‘消逝之火’则是一种珍贵但有限的能量，它来自古老的化石，藏在地球深处的煤炭、石油和天然气里。虽然‘消逝之火’可以燃烧出强大的力量，但它的力量是有限的，一旦用完，就再也不会回来了。\n"
    "所以，聪明的王国居民们开始越来越多地依靠‘永恒之光’，因为它既环保又永不枯竭，而‘消逝之火’虽然强大，但他们知道要珍惜使用。\n"
    "你看，可再生能源就像‘永恒之光’，而不可再生能源就像‘消逝之火’。记住哦，小朋友，我们要像王国的居民一样，学会利用那些永不耗尽的力量，让我们的世界变得更加美好！")
    ),
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

conv_bair_v1 = Conversation(
    system="BEGINNING OF CONVERSATION:",
    roles=("USER", "GPT"),
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)


default_conversation = conv_v1_2
conv_templates = {
    "v1": conv_v1_2,
    "bair_v1": conv_bair_v1,
}


if __name__ == "__main__":
    print(default_conversation.get_prompt())
