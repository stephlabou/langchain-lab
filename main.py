from constants import *
from util import load_config, RAG
import os

def main():
    config = load_config(configYamlPath)
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = config['HUGGINGFACEHUB_API_TOKEN']
    rag = RAG(config)
    rag(["https://ucsd.libguides.com/gis/gisdata"], use_strainer=True, strainer_class='s-lib-box-content')
    question = "What is X drive"
    print(rag.ask(question))
    
if __name__ == "__main__":
    main()