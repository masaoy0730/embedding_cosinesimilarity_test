from openai import AzureOpenAI
# クライアント作成
client = AzureOpenAI(azure_endpoint="https://<ターゲットURI参照>.openai.azure.com",
api_version="<バージョン>",
api_key="<apiキー>")

# Embeddingモデルを指定
model_id = "text-embedding-3-large" #利用するモデル名

# コサインや内積の計算のために必要
import numpy as np

# 2. テスト用の文章
texts = [
  "SharePoint Onlineの検索の事例を探してください",
  "また、クラウドストレージ（SharePoint Online）標準の検索機能では、全文検索や横断検索など、求めている十分な検索が行えなかったことも$
  "特に、現在はクラウドストレージへの移行期であるため、従業員が新旧どちらのストレージにファイルが保存されてるかを意識する必要もなく$
]

# 2-2. ラベル用
labels = ["プロンプト", "候補1", "候補2"]

# 3. 埋め込みベクトルの生成関数
def get_embedding(text):
    response = client.embeddings.create(input=[text],model = model_id)
    return response.data[0].embedding

# 各文章の埋め込みベクトルを取得
embeddings = [get_embedding(text) for text in texts]

# 4. コサイン類似度の計算関数
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 類似度の計算
similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])  # 文1と文2の類似度
similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])  # 文1と文3の類似度

# 5. 結果の表示
for label, text in zip(labels, texts):
 print(f"\n{label}:{text}")

# print(f"Similarity between Sentence 1 and Sentence 2: {similarity_1_2:.4f}")
# print(f"Similarity between Sentence 1 and Sentence 3: {similarity_1_3:.4f}")

print(f"\nSimilarity between プロンプト and 候補1: {similarity_1_2:.4f}")
print(f"Similarity between プロンプト and 候補2: {similarity_1_3:.4f}")
print(f"モデル:{model_id}")
