# FIXME hack
import sys
sys.path.insert(0, 'src')
from ai_security.chatter_demo import demo as chatter_demo
from ai_security.malware_demo import demo as malware_demo


# demo = chatter_demo
demo = malware_demo

if __name__ == "__main__":

    demo.launch()

