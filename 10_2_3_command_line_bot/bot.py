import boto3
import wikipedia

session = boto3.Session()
region = session.region_name

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    region_name=region,
)

modelId = "anthropic.claude-3-5-sonnet-20241022-v2:0"

toolConfig = {
    "tools": [
        {
            "toolSpec": {
                "name": "get_article",
                "description": "A tool to retrieve an up to date Wikipedia article.",
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "search_term": {
                                "type": "string",
                                "description": "The search term to find a wikipedia article by title",
                            }
                        },
                        "required": ["search_term"],
                    }
                },
            }
        }
    ]
}


def get_article(search_term):
    results = wikipedia.search(search_term)
    first_result = results[0]
    page = wikipedia.page(first_result, auto_suggest=False)
    return page.content


def answer_question(question):
    system_prompt = """
    You will be asked a question by the user. 
    If answering the question requires data you were not trained on, you must use the get_article tool to get the contents of a recent wikipedia article about the topic. 
    If you can answer the question without needing to get more information, please do so. 
    Only call the tool when needed. 
    """
    messages = [{"role": "user", "content": [{"text": question}]}]

    converse_api_params = {
        "modelId": modelId,
        "system": [{"text": system_prompt}],
        "messages": messages,
        "inferenceConfig": {"temperature": 0.0, "maxTokens": 4096},
        "toolConfig": toolConfig,  # The toolConfig which contains our article_search_tool details
    }

    response = bedrock_client.converse(**converse_api_params)

    while response["stopReason"] == "tool_use":
        tool_use = response["output"]["message"]["content"][-1]
        tool_id = tool_use["toolUse"]["toolUseId"]
        tool_name = tool_use["toolUse"]["name"]
        tool_inputs = tool_use["toolUse"]["input"]
        # Add Claude's tool use call to messages:
        messages.append(
            {"role": "assistant", "content": response["output"]["message"]["content"]}
        )

        if tool_name == "get_article":
            search_term = tool_inputs["search_term"]
            print(f"Claude wants to get an article for: {search_term}")
            wiki_result = get_article(search_term)  # get wikipedia article content
            # construct our tool_result message
            tool_response = {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": tool_id,
                            "content": [{"text": wiki_result}],
                        }
                    }
                ],
            }
            messages.append(tool_response)
            # respond back to Claude
            converse_api_params = {
                "modelId": modelId,
                "system": [{"text": system_prompt}],
                "messages": messages,
                "inferenceConfig": {"temperature": 0.0, "maxTokens": 4096},
                "toolConfig": toolConfig,  # The toolConfig which contains our article_search_tool details
            }

            response = bedrock_client.converse(**converse_api_params)

    print("Claude's final answer:")
    print(
        response["output"]["message"]["content"][0]["text"]
    )  # As of 2024 who has more oscars, christopher Nolan or Ben Stiller?


if __name__ == "__main__":
    print("Wikipedia Q&A Bot (type 'quit' or 'exit' to stop)")
    while True:
        question = input("\nEnter a question: ")
        if question.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        answer_question(question)
