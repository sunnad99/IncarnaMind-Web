import os
from fastapi import FastAPI, UploadFile, File, Depends
from fastapi import Query, Body

from chat_with_pdf import chat, load_retriever_chain
from convo_qa_chain import ConvoRetrievalChain
from docs2db import process_files

from params.chat_with_pdf_params import ChatWithPdfParams

from config import FS_PATH
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()


@app.post("/upload")
async def upload_file(user_id: str, document_id:str, file: UploadFile = File(...)):

    # Read file contents
    file_content: bytes = file.file.read()

    # Create the path to store the file
    root_dir: str = f"{FS_PATH}{user_id}/{document_id}"
    pdf_data_path: str = f"{root_dir}/data"
    pdf_file_path: str = f"{pdf_data_path}/{file.filename}"
    response: dict = {"message": "File stored successfully", "status": 200}
    try:

        # Check if the pdf data directory exists
        if not os.path.exists(pdf_data_path):
            os.makedirs(pdf_data_path)

        # Write the file to the pdf data directory for the respective user
        with open(pdf_file_path, "wb") as pdf_file:
            pdf_file.write(file_content)

            print(f"File {file.filename} stored successfully")

        # Process the pdfs, create the database and store the data
        is_processed: bool = process_files(root_dir)
        if not is_processed:
            response["message"] = "File stored successfully but could not be processed"
            response["status"] = 500

        return response
    except Exception as e:
        return {"error": str(e)}


@app.post("/chat")
async def chat_with_pdf(user_id: str, document_id: str, body: ChatWithPdfParams):

    # Read the parameters
    user_query: str = body.user_query
    chat_history: list = body.chat_history

    root_dir: str = f"{FS_PATH}{user_id}/{document_id}"
    # Check if the user directory exists
    if not os.path.exists(root_dir):
        return {"message": "User does not exist", "status": 400}

    # Check if the database exists
    db_path: str = f"{root_dir}/database_store"
    print(db_path)
    if not os.path.exists(db_path):
        return {"message": "User does not have a database", "status": 404}

    response = {"message": "", "status": 200}
    try:

        # Carry out the chat with the pdf
        retriever_chain: ConvoRetrievalChain = load_retriever_chain(db_path)
        ai_response: str = chat(user_query, chat_history, retriever_chain)
        response["message"] = ai_response

        return response
    except Exception as e:
        return {"error": str(e), "status": 500}
