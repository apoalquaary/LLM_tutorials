
'''
/**************************************************************
 *                                                            *
 *                     Done By: ApoAlquaary                   *
 *            Github: https://github.com/ApoAlquaary          *
 *     youtube: https://www.youtube.com/@AlquaaryAcademy      *
 *                     Date: 16/12/2023                       *
 * 				   	 							              *
 *************************************************************/
'''

## Libraries

from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch



# Model Name (locally or from hugging face)

# model = "meta-llama/Llama-2-7b-chat-hf"
model = "../../../LLMs/llama-2-7b-chat-hf"



# Tokenizer Object

tokenizer = AutoTokenizer.from_pretrained(model)



# pipeline object 

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)



# user query (Input)

print("\n\n\n\n\n\n")
query = "query"
while 1:


    query = input("Enter a Query:   ")
    if query.lower() == "exit":
        break


    # Generating Output
    sequences = pipeline(
        query,
        max_length=200, # 
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
    )


    print("\n\n\n\n\n\n")
    # printing the output
    for seq in sequences:
        print(f"Result: {seq['generated_text']}")
        print("*******************************")

