import os # 環境変数の操作
from dotenv import load_dotenv # .envファイルを読み込み、その内容を環境変数として設定するために使用
from openai import AzureOpenAI #openaiのapiを操作
import numpy as np #ベクトルの計算

# .envファイルの内容を読み込み
load_dotenv(dotenv_path="/Users/yanagisawamasao/python/myenv/embedding_test/notebooks/.env")

# 環境変数からAPIキーとエンドポイントを取得
api_key = os.getenv('AZURE_OPENAI_KEY')
endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')

# Embeddingモデルを指定
model_id = "text-embedding-3-large"
#model_id = "text-embedding-3-small" 
version="2023-05-15"

# クライアント作成
client = AzureOpenAI(azure_endpoint=endpoint,
api_version=version,
api_key=api_key)

# テスト用の文章
texts = [
  "SharePoint Onlineの検索の事例を探してください",
  "また、クラウドストレージ（SharePoint Online）標準の検索機能では、全文検索や横断検索など、求めている十分な検索が行えなかったこともあり、それらを解決する精度の高い検索システムを探す必要がありました。特に技術文書などの検索に重きを置いている、設計・施工部門からの要望もあり、課題解決のためのツールの検討を開始しました。1.取り組みの背景 - Neuron ES導入後の評価 - ”仕事のテンポを止めない” 検索ストレスが低減し、業務効率UP！◆ クラウドストレージへの移行に伴いデータの所在が分散したが、検索システムによってそれを意識することなく欲しい情報に辿り着けている◆ 自社システムとのAPI連携により、従業員に新たな利用価値創出導入事例：大和ハウス工業株式会社SharePoint Onlineの検索性向上にNeuron ESを導入。さらに既存の業務システムとのAPI連携によって、新たな利用価値を創出。大和ハウス工業株式会社",
  "特に、現在はクラウドストレージへの移行期であるため、従業員が新旧どちらのストレージにファイルが保存されてるかを意識する必要もなく、しかもスピーディに目的のファイルを見つけられるようになったため、以前と比べて利便性自体も高くなったと感じています。また、SharePoint Online標準の検索では精度が不十分であった全文検索にも対応したので、社内規定やマニュアル、申請書といった書類を検索するのにも役立っています。全文検索：ファイル内のすべての文章を対象とする検索のこと社内業務システムともAPI連携を行い検索を軸とした新たな可能性や価値を創出今回の導入では、SharePoint Online上の検索性能を単に向上させただけでなく、従業員への通達（連絡）に利用する社内業務システムを刷新する際に「Neuron Enterprise Search」を検索エンジンとして組み込みました。これによって検索精度が向上しただけではなく、添付ファイルの全文検索も可能となりました。こうしたAPI連携の機能が提供されていることで、検索を軸とした新たな可能性や価値の創出ができるのも面白いと感じています"
]

# ラベル用
labels = ["プロンプト", "候補1", "候補2"]

# 埋め込みベクトルの生成関数
def get_embedding(text):
    response = client.embeddings.create(input=[text],model = model_id)
    return response.data[0].embedding

# 各文章の埋め込みベクトルを取得
embeddings = [get_embedding(text) for text in texts]

# コサイン類似度の計算関数
def cosine_similarity(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# 類似度の計算
similarity_1_2 = cosine_similarity(embeddings[0], embeddings[1])  # 文1と文2の類似度
similarity_1_3 = cosine_similarity(embeddings[0], embeddings[2])  # 文1と文3の類似度

# 結果の表示
for label, text in zip(labels, texts):
 print(f"\n{label}:{text}")

print(f"\nSimilarity between プロンプト and 候補1: {similarity_1_2:.4f}")
print(f"Similarity between プロンプト and 候補2: {similarity_1_3:.4f}")
print(f"モデル:{model_id}")
print(f"次元：{len(embeddings[0])}")
embeddings = [get_embedding(text) for text in texts]
print(f"サンプルベクトル：{embeddings[0][:5]}")
