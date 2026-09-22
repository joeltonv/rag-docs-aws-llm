import boto3

def testar_conexao():
    try:
        print("Iniciando teste de conexão com Amazon Bedrock...")
        client = boto3.client('bedrock-runtime', region_name='us-east-1')

        response = client.converse(
            modelId='deepseek.v3.2',
            messages=[{'role': 'user', 'content': [{'text': 'Responda apenas "Conexão bem-sucedida!".'}]}],
            inferenceConfig={'maxTokens': 50, 'temperature': 0.1}
        )

        texto_resposta = response['output']['message']['content'][0]['text']
        print(f"✅ SUCESSO! Resposta do LLM: {texto_resposta}")

    except Exception as e:
        print(f"❌ ERRO DE CONEXÃO:\n{str(e)}")

if __name__ == "__main__":
    testar_conexao()
