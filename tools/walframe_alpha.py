"""
    微分计算器
"""
import os
from langchain.utilities import wolfram_alpha
os.environ['WOLFRAM_ALPHA_APPID'] = '6VTLJQ-EAPJ3TXKGH'
wolfram = wolfram_alpha.WolframAlphaAPIWrapper()

print(wolfram.run("What is 2x+5 = -3x + 20?"))
