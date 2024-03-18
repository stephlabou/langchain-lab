To launch the application, run `python app.py`

## Requirements for Launching Application
This project uses models from HuggingFace. For the application to launch, **HuggingFace API token is required** which can be found in `config.yaml`. [Check this guide to create HuggingFace API token](https://huggingface.co/docs/hub/en/security-tokens) 

## Project Description
This project aims to explore the use of Retrieval-Augmented Generation (RAG) on UC San Diego's Library Page to enhance searching experience. All models used in this project are pretrained models from HuggingFace and some utilities are from LangChain.

## Files
The following files are important for launching the application:

- `app.py` - The Flask application file. To launch, run `python app.py`
- `templates` - HTML and JavaScript can be found here
- `static` - CSS can be found here
- `config/config.yaml` - Configuration used to run the `app.py` file. This is mainly used to define parameters and token for the model.

## Reference
- https://python.langchain.com/docs/use_cases/question_answering/quickstart
