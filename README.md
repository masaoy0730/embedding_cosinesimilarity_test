# embedding_cosine_similarity_test
RAGの実験のために、ベクトル化とコサイン類似度の計算をテストしました。<br>

サンプルの文書をベクトル化し、コサイン類似度を計算するテストです。<br>
pythonを使っています。EmbeddingのモデルはAzure OpenAI Serviceのtext-embedding-3-largeとtext-embedding-3-smallです。<br>
PythonのOpenAIのAPIを使うためのライブラリが新しくなっており、関数や引数の名前で少し苦労したので、メモを残します。<br>
openaiのライブラリはVersion: 1.70.0です。

## 結果
text-embedding-3-largeを使った時、コサイン類似度を計算すると次のような結果が得られます。

|ベクトル1|ベクトル2|値|
|:------|:-------|:--|
|プロンプト|候補1|0.5854|
|プロンプト|候補2|0.6143|

プロンプト、および、類似度を計算した文書は下記です。

|種類|内容|
|:--|:---|
|プロンプト|SharePoint Onlineの検索の事例を探してください|
|候補1|また、クラウドストレージ（SharePoint Online）標準の検索機能では、全文検索や横断検索など、求めている十分な検索が行えなかったこともあり、それらを解決する精度の高い検索システムを探す必要がありました。特に技術文書などの検索に重きを置いている、設計・施工部門からの要望もあり、課題解決のためのツールの検討を開始しました。1.取り組みの背景 - Neuron ES 導入後の評価 - ”仕事のテンポを止めない” 検索ストレスが低減し、業務効率UP！ ◆ クラウドストレージへの移行に伴いデータの所在が分散したが、検索システムによってそれを意識することなく欲しい情報に辿り着けている◆ 自社システムとのAPI連携により、従業員に新たな利用価値創出 導入事例：大和ハウス工業株式会社SharePoint Onlineの検索性向上にNeuron ESを導入。さらに既存の業務システムとのAPI連携によって、新たな利用価値を創出。大和ハウス工業株式会社|
|候補2|特に、現在はクラウドストレージへの移行期であるため、従業員が新旧どちらのストレージにファイルが保存されてるかを意識する必要もなく、しかもスピーディに目的のファイルを見つけられるようになったため、以前と比べて利便性自体も高くなったと感じています。また、SharePoint Online標準の検索では精度が不十分であった全文検索にも対応したので、社内規定やマニュア ル、申請書といった書類を検索するのにも役立っています。全文検索：ファイル内のすべての文章を対象とする検索のこと社内業務システムともAPI連携を行い 検索を軸とした新たな可能性や価値を創出今回の導入では、SharePoint Online上の検索性能を単に向上させただけでなく、従業員への通達（連絡）に利用する社内業務システムを刷新する際に「Neuron Enterprise Search」を検索エンジンとして組み込みました。これによって検索精度が向上しただけではなく、添付ファイルの全文検索も可能となりました。こうしたAPI連携の機能が提供されていることで、検索を軸とした新たな可能性や価値の創出ができるのも面白いと感じています|


構築した環境とコードが下記になります。

## 0.環境設定
仮想環境を使います。仮想環境を使わない方は1.をご覧ください。<br>
新しい環境を構築します。
```bash
python -m venv <新しい仮想環境ディレクトリ>
```
仮想環境有効化します。
```
source bin/activate
```

### 1.必要なモジュールをインストール
openaiのモジュールをインストールします。<br>
類似度計算でcosineやベクトルの内積を計算するのでnumpyも必要です。
```bash
pip install openai numpy
```
## 2. Azure OpenAI ServiceでAPIを用意
Azure PortalからAzure AI Foundryを開き、text-embedding-3-largeとtext-embedding-3-smallをデプロイします。<br>
uri、キーなどを使います。

## 3.サンプルプログラムの作成
プログラムは[embedding.py](./embedding.py)をご覧ください。<br>
コードで苦労したところをメモします。

### 3-1.openaiのライブラリの読み込み
`import openai`ではエラーになったので、下記にします。
```
from openai import AzureOpenAI
```
### 3-2.クライアント作成
Azure AI Foundryでモデルをデプロイした画面から、ターゲットURIを参照し、エンドポイントとバージョン、apiキーを記載します。
```
client = AzureOpenAI(azure_endpoint="https://<ターゲットURI参照>.openai.azure.com",
api_version="<バージョン>",
api_key="<apiキー>")
```
### 3-3. Embeddingモデルを指定
deployment_idでの指定がうまくいかなかったので、`model_id`にモデル名を入力することで通りました。
```
model_id = "text-embedding-3-large" #利用するモデル名
```

### 3-4. 埋め込みベクトルの生成関数
作成したclientを使うようにします。
```
def get_embedding(text):
    response = client.embeddings.create(input=[text],model = model_id)
    return response.data[0].embedding
```

## 4.出力例
embedding.pyの実行後の出力例です。
```
Similarity between プロンプト and 候補1: 0.5854
Similarity between プロンプト and 候補2: 0.6143
```

## 6.比較
largeとsmallで比較します。<br>
<br>

モデル：text-embedding-3-large
|ベクトル1|ベクトル2|値|
|:------|:-------|:--|
|プロンプト|候補1|0.5854|
|プロンプト|候補2|0.6143|

モデル:text-embedding-3-small
|ベクトル1|ベクトル2|値|
|:------|:-------|:--|
|プロンプト|候補1|0.6175|
|プロンプト|候補2|0.6183|

<br>
どちらのモデルでも0.7以下の値。smallの方が若干値が高い。

## 7.追加テスト
Difyで検索テストした結果を記載します。同じプロンプトに基づいてナレッジのヒット効果をテストしました。候補1と候補2のスコアは下記です。

### 条件1：ベクトル検索のみ
ハイブリッド検索の設定はウェイト設定にし、セマンティックマッチングを1にします。
|ベクトル1|ベクトル2|スコア|
|:------|:-------|:--|
|プロンプト|候補1|0.59|
|プロンプト|候補2|0.59|

スコアはコサイン類似度の値と近い数字になっています。

### 条件2：ハイブリッド検索
ハイブリッド検索の設定はウェイト設定にし、セマンティックマッチング0.7, キーワードマッチング0.3にします。
|ベクトル1|ベクトル2|スコア|
|:------|:-------|:--|
|プロンプト|候補1|0.47|
|プロンプト|候補2|0.44|

キーワードマッチング入れることでスコアは全体的に下がりました。

### 条件３：ハイブリッド検索（リランク利用）
ハイブリッド検索の設定はRerank設定にし、cohereのrerank-multilingual-v3.0を利用します。
|ベクトル1|ベクトル2|スコア|
|:------|:-------|:--|
|プロンプト|候補1|0.99|
|プロンプト|候補2|0.44|

上記の結果から、Rerankモデルを利用することでスコアが大きくなるチャンクが出てくることがわかりました。
回答精度向上につながると考えられますが、また別の実験を検討したいと思います。


## 8.付録
実際に「SharePoint Onlineの検索の事例を探してください」をEmbeddingした結果を下記に示します。
<!-- open属性なし -->
<details><summary>Embeddingのサンプル</summary>
```
[-0.011686011217534542, -0.004690196365118027, -0.010317379608750343, 0.05402937904000282, 0.010733232833445072, -0.030383629724383354, -0.009559367783367634, 0.03834275156259537, 0.01705526001751423, 0.002018732251599431, 0.03945871442556381, -0.06080937013030052, 0.0216454416513443, -0.0391007624566555, -0.00037110981065779924, 0.03686884045600891, 0.0006063434411771595, -0.03476325049996376, -0.015949826687574387, 0.04213280975818634, 0.0031794372480362654, -0.015233926475048065, 0.009938374161720276, 0.02914133295416832, -0.020782150328159332, -0.048344291746616364, 0.022319229319691658, -0.002071371767669916, 0.01931876689195633, -0.007243221625685692, 0.01903451234102249, -0.03307878226041794, 0.02332991175353527, 0.05179745703935623, 0.00450595747679472, 0.03168909251689911, -0.020782150328159332, 0.005453472025692463, -0.002624088665470481, 0.00012921399320475757, 0.010464770719408989, 0.03320511803030968, 0.0445963479578495, 0.03990088775753975, 0.0691474974155426, -0.0030109903309494257, -0.018434420228004456, -0.04665982350707054, 0.02610928751528263, -0.01260194182395935, 0.008390766568481922, -0.027246303856372833, -0.0028056956361979246, -0.06855793297290802, -0.00027241039788350463, -0.02158227376639843, 0.009601479396224022, 0.01499178446829319, 0.03023623861372471, 0.013644208200275898, -0.007506419904530048, -0.0007777514401823282, -0.01789749413728714, 0.044470012187957764, 0.013054643757641315, 0.0214875228703022, -0.01190709788352251, 0.02025575377047062, -0.021213795989751816, 0.01139122899621725, 0.013233618810772896, 0.019824108108878136, 0.015118119306862354, 0.029941456392407417, -0.003590026870369911, -0.003992720507085323, 0.021403297781944275, 0.015823490917682648, -0.0032057571224868298, -0.04061678424477577, 0.012275575660169125, -0.007169526070356369, -0.01266510970890522, -0.016244608908891678, 0.003782161744311452, 0.02884655073285103, 0.0010521358344703913, 0.018845008686184883, -0.018434420228004456, 0.014844393357634544, -0.003742682049050927, 0.017171068117022514, 0.01121225394308567, -0.008027552627027035, 0.03076263517141342, 0.04842851683497429, -0.02526705153286457, -0.012938836589455605, -0.00790121778845787, 0.051418449729681015, 0.032741889357566833, -0.08477096259593964, 0.007701186928898096, -0.022929849103093147, -0.02387736365199089, -0.060009248554706573, 0.02836226485669613, 0.0011074075009673834, 0.012170297093689442, -0.019402990117669106, -0.12818817794322968, -0.029878288507461548, -0.05925123766064644, 0.04703882709145546, 0.013644208200275898, -0.014107436873018742, 0.03579498827457428, -0.013517872430384159, -0.01363367959856987, 0.0206768698990345, -0.01050161849707365, 0.026509348303079605, -0.024319536983966827, 0.01214924082159996, 0.016981564462184906, 0.03293139114975929, -0.03379468247294426, -0.02998356893658638, -0.03809008002281189, -0.008369711227715015, 0.038658589124679565, 0.019402990117669106, 0.029057109728455544, -0.027604253962635994, 0.016486750915646553, 0.0033768361900001764, -0.03834275156259537, 0.0008047292940318584, -0.056429747492074966, 0.009901526384055614, 0.006048300303518772, -0.048512738198041916, 0.004274343140423298, 0.04013250023126602, -0.054618943482637405, -0.012886196374893188, 0.014718057587742805, -0.04733360931277275, 0.006995814852416515, -0.04126951843500137, -0.022403452545404434, -0.07255855202674866, -0.045017462223768234, 0.012201880104839802, -0.01308622770011425, -0.011380701325833797, -0.00563771091401577, -0.01796066202223301, -0.012538774870336056, -0.004753364250063896, 0.047670505940914154, 0.0164762232452631, -0.01592877134680748, 0.025119660422205925, -0.0031189017463475466, 0.02610928751528263, 0.022150782868266106, 0.017265819013118744, -0.060725148767232895, 0.004992874804884195, -0.02533021941781044, -0.061272602528333664, 0.011201726272702217, 0.04737572371959686, 0.04015355557203293, 0.015233926475048065, 0.020476840436458588, -0.019539853557944298, 0.00519290566444397, 0.012170297093689442, -0.011422812938690186, 0.010069972835481167, -0.005303448997437954, -0.01474964153021574, -0.015149703249335289, -0.0240668673068285, -0.02297196164727211, 0.012286104261875153, 0.0035716029815375805, -0.016097217798233032, 0.0164762232452631, -0.0015318150399252772, 0.020561063662171364, -0.0013752119848504663, 0.00035992389894090593, -0.037142567336559296, -0.03314194828271866, -0.00598513288423419, -0.010022597387433052, -0.008627645671367645, -0.011433340609073639, -0.015360262244939804, -0.03547915071249008, 0.015612932853400707, 0.0034110520500689745, -0.00790121778845787, 0.005674558691680431, -0.00886978767812252, -0.034068409353494644, 0.001908188802190125, 0.05731409415602684, 0.03362623602151871, 0.02895183116197586, -0.028214873746037483, 0.03657405823469162, -0.017076315358281136, -0.0016792061505839229, 0.001804225379601121, 0.008180207572877407, -0.005000770557671785, -0.019055567681789398, 0.014560138806700706, -0.016044577583670616, 0.010275267995893955, 0.008180207572877407, 0.06969495117664337, 0.05647186189889908, -0.06312552094459534, 0.012980948202311993, -0.07251644134521484, 0.013549456372857094, 0.027056802064180374, -0.015949826687574387, 0.008617117069661617, 0.037563685327768326, 0.014581194147467613, 0.003966400865465403, -0.009991013444960117, 0.0024661696515977383, -0.005695614498108625, -0.025372331961989403, -0.051123667508363724, 0.0300467349588871, 0.015539237298071384, 0.06889482587575912, -0.012401911430060863, 0.013065171428024769, -0.008474990725517273, 0.02461431920528412, 0.03750051558017731, 0.0010310799116268754, -0.0016568342689424753, 0.014939144253730774, 0.014128493145108223, -0.0011936050141230226, -0.03379468247294426, -0.048765409737825394, 0.008743452839553356, -0.0028899190947413445, -0.0015423429431393743, 0.000394468690501526, -0.00504025025293231, -0.007953857071697712, -0.033710457384586334, -0.08632909506559372, -0.009085610508918762, 0.03324722871184349, -0.018666034564375877, -0.0015594508731737733, 0.0004089446156285703, -0.006874743849039078, 0.02981512062251568, 0.033752571791410446, -0.005442943889647722, -0.019402990117669106, -0.005490319803357124, -0.01761324144899845, -0.0345737487077713, -0.02684624306857586, -0.009943637996912003, -0.04417522996664047, 0.013412592932581902, -0.008880316279828548, 0.02332991175353527, 0.05445049703121185, 0.007011606823652983, 0.065273217856884, -0.02465643174946308, -0.07538003474473953, 0.027667421847581863, -0.026825185865163803, -0.0027714797761291265, -0.016128800809383392, -0.0261724554002285, -0.034910641610622406, 0.034321077167987823, -0.024530095979571342, -0.026572516188025475, 0.016949981451034546, 0.053229257464408875, 0.009411976672708988, -0.016876285895705223, 0.056429747492074966, 0.03082580305635929, 0.04594392329454422, 0.013191506266593933, -0.0011041174875572324, -0.010754289105534554, 0.04042728245258331, -0.016728894785046577, -0.00861185323446989, -0.012791444547474384, 0.029057109728455544, -0.019887275993824005, 0.013402065262198448, -0.045564915984869, 0.06388352811336517, 0.026088230311870575, 0.02394053153693676, -0.05289236083626747, 0.027541086077690125, 0.01614985801279545, -0.011854458600282669, -0.029288724064826965, -0.01958196610212326, 0.0020805837120860815, -0.01602352224290371, 0.016981564462184906, 0.00938565656542778, -0.02177177555859089, -0.007616963237524033, 0.012928307987749577, -0.024151090532541275, 0.015655044466257095, -2.407854663033504e-05, -0.053776707500219345, -0.012359799817204475, 0.03421580046415329, -0.03259449824690819, -0.009638327173888683, -0.04986031726002693, -0.015402373857796192, -0.0067168246023356915, -0.06923171877861023, -0.0016897340537980199, -0.03251027315855026, -0.01151756476610899, -0.007753826677799225, -0.02206655777990818, -0.004190119449049234, 0.016876285895705223, -0.0020174160599708557, 0.008017024956643581, 0.023729972541332245, 0.04539646953344345, -0.04413311555981636, -0.011654427275061607, -0.02707785740494728, 0.033584121614694595, -0.004379622172564268, -0.03632138669490814, 0.0064220423810184, -0.006648392882198095, -0.03114164061844349, -0.0044427900575101376, 0.04354355111718178, -0.02364574931561947, -0.030067792162299156, 0.0755905956029892, 0.012612470425665379, -0.0024516936391592026, 0.042701318860054016, -0.02646723762154579, 0.010201572440564632, -0.02587767317891121, -0.021182211115956306, 0.006616809405386448, 0.04720727354288101, -0.002145067322999239, 0.023287799209356308, 0.002060843864455819, -0.05023932084441185, -0.04872329905629158, 0.008396030403673649, 0.004000616725534201, 0.007395876571536064, -0.01903451234102249, 0.007874897681176662, -0.0008606589399278164, -0.007959120906889439, -0.013465233147144318, 0.0168025903403759, 0.03964821621775627, -0.005342928692698479, 0.009796246886253357, 0.0015028631314635277, 0.04139585420489311, 0.015212871134281158, 0.003568971063941717, -0.008506573736667633, -0.025477610528469086, 0.015844546258449554, -0.029709842056035995, 0.03627927601337433, 0.0074116685427725315, 0.0011692591942846775, -0.03347884491086006, -0.012001849710941315, 0.01411796547472477, -0.036237165331840515, -0.02726735919713974, -0.02937294729053974, 0.013170450925827026, -0.010122613050043583, 0.04396466910839081, 0.003840065561234951, -0.010185780003666878, 0.1098274514079094, 0.0015897186240181327, -0.016002466902136803, -0.010701648890972137, -0.004142743535339832, 0.034973811358213425, 0.03147853538393974, -0.007948593236505985, 0.01100169587880373, 0.06531532853841782, -0.020634759217500687, 0.07024240493774414, -0.01532867830246687, -0.035331759601831436, 0.006485209800302982, 0.02316146343946457, 0.01622355356812477, -0.04055361822247505, -0.04164852201938629, -0.018813425675034523, -0.004066416062414646, 0.0007428776589222252, 0.024045810103416443, -0.03771107643842697, -0.017876438796520233, -0.0023964219726622105, 0.014665418304502964, -0.04333299398422241, -0.03562654182314873, -0.006079884245991707, -0.009538312442600727, -0.06413619965314865, 0.012286104261875153, 0.0285938810557127, -0.00015693207387812436, 0.022656124085187912, 0.04215386509895325, 0.0038795452564954758, -0.06400986760854721, -0.021371714770793915, -0.07706450670957565, 0.027183135971426964, 0.013170450925827026, -0.015676099807024002, 0.02236134000122547, 0.0024516936391592026, 0.020624231547117233, 0.013465233147144318, -0.005853533744812012, -0.020813733339309692, -0.01022262778133154, 0.002766215708106756, 0.010069972835481167, 0.02979406528174877, 0.048933856189250946, 0.05596651881933212, -0.019013457000255585, -0.027225248515605927, 0.00034873795812018216, -0.002046368084847927, -0.03613188490271568, -0.03137325495481491, -0.053355589509010315, -0.015970882028341293, -0.028635991737246513, 0.0028793911915272474, 0.039627160876989365, -0.004987610969692469, 0.14587511122226715, 0.025309164077043533, 0.017076315358281136, -0.02840437740087509, -0.008111776784062386, -0.023856308311223984, 0.05301869660615921, -0.003761105937883258, -0.006864215712994337, -0.022024447098374367, -0.03282611072063446, -0.0083170710131526, -0.03947976976633072, -0.008069664239883423, 0.03259449824690819, -0.0216454416513443, -0.049186527729034424, -0.03695306181907654, 0.014886504970490932, 0.021150628104805946, -0.04998664930462837, 0.02212972566485405, -0.004411206115037203, 0.004924443084746599, -0.002683308208361268, 0.03276294469833374, 0.017665879800915718, 0.010164724662899971, 0.015002312138676643, 0.008696077391505241, -0.0033663082867860794, -0.012875668704509735, 0.0025135453324764967, -0.0019897802267223597, -0.03288928046822548, -0.020245226100087166, 0.005495583638548851, -0.0016318304697051644, 0.005071834195405245, -0.01589718647301197, 0.002787271747365594, -0.00554822338744998, -0.024172145873308182, -0.014170604757964611, -0.03451058268547058, -0.01033843494951725, -0.008911899290978909, -0.004821795504540205, 0.010780609212815762, -0.001421271706931293, -0.018550226464867592, 0.006980022881180048, 0.004813899751752615, -0.0021055876277387142, 0.04219597578048706, 0.02019258588552475, 0.010764816775918007, -0.01432852353900671, 0.0352054238319397, -0.004874435253441334, -0.03284716606140137, 0.02045578323304653, -0.0003586079110391438, -0.01992938667535782, 0.03924815356731415, -0.012823028489947319, -0.032615553587675095, 0.0035716029815375805, -0.0064378343522548676, 0.0006362822605296969, 0.01621302403509617, 0.005695614498108625, -0.000959358352702111, -0.018371252343058586, -0.017792215570807457, -0.012675637379288673, -0.0014975991798564792, 0.0010889836121350527, -0.007016870658844709, -0.02083479054272175, -0.020592646673321724, -0.0022608747240155935, -0.01928718201816082, -0.02351941354572773, -0.003653194522485137, -0.039269208908081055, 0.019181903451681137, -0.007216901518404484, 0.06531532853841782, -0.029520338401198387, -0.015139175578951836, 0.0034979074262082577, -0.017750103026628494, 0.011138558387756348, -0.016128800809383392, 0.01708684302866459, -0.03387890383601189, -0.039500825107097626, 0.01925559900701046, -0.016497278586030006, -0.0691474974155426, 0.0361950509250164, 0.02490910142660141, 0.06320974230766296, 0.01896081678569317, 0.003547915257513523, -0.004827059805393219, 0.037142567336559296, -0.03139431029558182, 0.040532562881708145, -0.02242450788617134, 0.051292113959789276, -0.0004523723619058728, 0.024635376408696175, -0.015465541742742062, -0.015949826687574387, -0.03661616891622543, 0.011528092436492443, -0.031710151582956314, 0.022445565089583397, 0.017718520015478134, 0.02539338730275631, 0.006743144243955612, 0.03773213177919388, 0.008975067175924778, -5.551842718887201e-07, 0.015791907906532288, 0.004971818998456001, -0.01977146789431572, 0.03947976976633072, -0.012433495372533798, -0.026867298409342766, -0.008438142947852612, 0.025309164077043533, 0.012180824764072895, -0.03173120692372322, -0.03705834224820137, 0.03893231600522995, 0.02985723316669464, 0.020855845883488655, -0.00855921395123005, 0.002227974822744727, -0.021624384447932243, -0.02872021496295929, -0.013223090209066868, -0.0030004624277353287, -0.018086997792124748, 0.012928307987749577, -0.04956553503870964, 0.012570357881486416, -0.03276294469833374, -0.01393899042159319, 0.05554540082812309, -0.008727660402655602, 0.011980793438851833, 0.00790121778845787, 0.015591876581311226, 0.035268593579530716, -0.05078677460551262, 0.0260040070861578, 0.0552506186068058, -0.0018779209349304438, 0.004882331471890211, -0.028909718617796898, -0.005421888083219528, -0.025793448090553284, 0.0009481724118813872, 0.0011008275905624032, 0.03046785295009613, 0.009506728500127792, -0.017276346683502197, -0.018181748688220978, -0.006048300303518772, -0.02884655073285103, -0.00919089000672102, -0.004637556616216898, 0.032973501831293106, -0.028551768511533737, -0.015981409698724747, -0.02749897539615631, 0.014381163753569126, 0.023287799209356308, 0.022993016988039017, 0.01100169587880373, 0.046828269958496094, -0.04535435885190964, 0.011707067489624023, 0.025245996192097664, -0.011896570213139057, -0.004821795504540205, -0.014707529917359352, 0.0033136685378849506, -0.02368786185979843, 0.03305772691965103, 0.0012541406322270632, -0.01916084811091423, 0.0159919373691082, -0.04964975640177727, -0.034615859389305115, 0.0006626021349802613, 0.00039677167660556734, -0.004705988336354494, 0.026572516188025475, -0.006427306216210127, 0.02775164507329464, 0.018718674778938293, -0.028067482635378838, 0.017876438796520233, -0.034131575375795364, -0.013949518091976643, 0.007601171266287565, 0.0008363131200894713, -0.013044115155935287, 0.0025188091676682234, 2.7635838705464266e-05, 0.017939606681466103, 0.04830218106508255, -0.019908331334590912, -0.016960509121418, 0.005769310053437948, -0.008501309901475906, -0.033099837601184845, -0.01172812283039093, 0.032068099826574326, -0.030783692374825478, 0.02351941354572773, -0.015676099807024002, -0.029709842056035995, -0.0013304682215675712, 0.035394929349422455, 0.020213641226291656, -0.01614985801279545, -0.008985595777630806, -0.0032689247746020555, 0.004113791976124048, 0.024572208523750305, 0.0072274296544492245, -0.01931876689195633, -0.06438887119293213, -0.041753802448511124, -0.02368786185979843, 0.017950134351849556, 0.012759861536324024, -0.006306235212832689, -0.009875206276774406, -0.01822386123239994, 0.01103327888995409, -0.018497588112950325, 0.0005395568441599607, 0.010069972835481167, 0.0300467349588871, 0.01232821587473154, 0.03621610626578331, -0.003374204272404313, -0.03349990025162697, 0.001685786060988903, 0.03903759643435478, -0.003882177174091339, 0.0017160539282485843, -0.01260194182395935, -0.013949518091976643, -0.0035215953830629587, 0.025288106873631477, -0.056977201253175735, 0.0008718448807485402, 0.015297094359993935, 0.0038137456867843866, 0.028151707723736763, 0.007648547179996967, 0.03204704448580742, -0.015191814862191677, -0.016276191920042038, 0.02112957276403904, 0.03337356448173523, -0.028867606073617935, 0.030509965494275093, 0.03556337580084801, -0.00901191495358944, 0.01906609535217285, 0.01318097859621048, -0.035647597163915634, -0.013591567985713482, -0.014833864755928516, -0.030152015388011932, 0.005490319803357124, -0.010496354661881924, -0.019371407106518745, -0.0037874258123338223, 0.01605510525405407, 0.048765409737825394, 0.0465334877371788, -0.03697412088513374, -0.015686627477407455, -0.016528863459825516, -0.005769310053437948, 0.017213178798556328, 0.02413003519177437, 0.010464770719408989, 0.014160077087581158, -0.009659383445978165, -0.00399798434227705, 0.012433495372533798, -0.02962561883032322, -0.0313311442732811, -0.0193819347769022, 0.03379468247294426, 0.007816994562745094, 0.03550020605325699, -0.023856308311223984, 0.008427614346146584, 0.014433803036808968, -0.02787798084318638, -0.006485209800302982, -0.03954293578863144, 0.01082272082567215, -0.029667729511857033, 0.009490936063230038, -0.00979098305106163, 0.02787798084318638, -0.019960971549153328, 0.012622998096048832, 0.020961124449968338, -0.01883448101580143, -0.03971138224005699, -0.004411206115037203, -0.002012152224779129, -0.024698542430996895, -0.010291059501469135, 0.015918241813778877, -0.03366834670305252, 0.015918241813778877, 0.02981512062251568, 0.011570204049348831, -0.020792677998542786, 0.02646723762154579, 0.031457480043172836, -0.006322026718407869, -0.03590026870369911, -0.027414752170443535, -0.004282238893210888, 0.016655199229717255, -0.011159614659845829, 0.035394929349422455, 0.032194435596466064, 0.0014607514021918178, -0.015655044466257095, 0.007785410154610872, -0.03737417981028557, -0.01673942245543003, 0.0285938810557127, -0.008190736174583435, 0.008743452839553356, -0.013075699098408222, -0.024866990745067596, -0.0024872254580259323, -0.001681838184595108, -0.009601479396224022, -0.020297864452004433, -0.025014381855726242, -0.029036054387688637, -0.013549456372857094, -0.0001020716517814435, 0.04964975640177727, -0.03368940204381943, -0.0057429904118180275, 0.03476325049996376, 0.02956245094537735, 0.018666034564375877, 0.01040160283446312, -0.0029951983597129583, 0.030488910153508186, -0.007674866821616888, -0.01625513657927513, -0.02985723316669464, 0.011370173655450344, -0.014496970921754837, 0.0069221192970871925, -0.008180207572877407, 0.03242604807019234, -0.0008527630125172436, -0.02364574931561947, 0.021898111328482628, 0.02962561883032322, -0.023729972541332245, -0.03095213882625103, 0.01360209658741951, 0.011833402328193188, -0.02465643174946308, 0.0004237495013512671, -0.013096755370497704, -0.0005727856187149882, -0.002968878485262394, 0.03537387400865555, -0.07158997654914856, 0.037332069128751755, 0.03897442668676376, -0.031457480043172836, -0.00996469333767891, 0.03189965337514877, 0.018339669331908226, -0.009638327173888683, -0.05381882190704346, 0.03368940204381943, -0.03023623861372471, -0.012170297093689442, 0.002460905583575368, -0.01948721334338188, -0.013970574364066124, -0.05689297989010811, -0.035689711570739746, 0.032973501831293106, 0.03181542828679085, 0.03253132849931717, 0.006485209800302982, 0.0300467349588871, 6.612861034227535e-05, 0.04078523442149162, 0.0336051806807518, 0.06240961700677872, -0.009764662943780422, 0.017434265464544296, 0.025772392749786377, 0.0227824579924345, -0.01909768022596836, 0.030573133379220963, -0.00033722302759997547, 0.002420109696686268, 0.0028714952059090137, 0.010138404555618763, 0.0016318304697051644, -0.03882703557610512, 0.016076162457466125, 0.02290879376232624, -0.01657097414135933, 0.02039261721074581, -0.019360877573490143, -0.028383322060108185, 0.0504077672958374, -0.008822412230074406, 0.014202188700437546, 0.0063904584385454655, -0.018592339009046555, 0.03870069980621338, 0.030362574383616447, 0.027962204068899155, 0.004803371615707874, -0.009143514558672905, 0.00667997682467103, 0.03253132849931717, 0.042764484882354736, -0.027920091524720192, 0.03535281494259834, 0.012696693651378155, -0.013338897377252579, -0.0041585355065763, 0.016792060807347298, 0.022235006093978882, -0.013381009921431541, -0.006948439404368401, -0.02003466710448265, -0.04804950952529907, -0.0008435510681010783, 0.0005471237818710506, 0.009454088285565376, -0.02316146343946457, 0.002729367930442095, -0.01777116023004055, 0.005427152384072542, -0.032068099826574326, -0.06093570590019226, 0.017076315358281136, -0.020508423447608948, -0.011022751219570637, 0.028551768511533737, 0.027309471741318703, -0.023898418992757797, 0.046407151967287064, 0.012096600607037544, 0.005490319803357124, 0.0026675164699554443, 0.019760940223932266, 0.034552693367004395, -0.015360262244939804, 0.013802126981317997, -0.0008422350510954857, 0.029351891949772835, 0.01964513212442398, -0.01997149921953678, 0.003016254398971796, -0.06704191118478775, -0.013654735870659351, 0.04598603397607803, -0.006506266072392464, 0.013265201821923256, -0.019813580438494682, -0.017729047685861588, 0.03141536936163902, -0.045312248170375824, -0.033226173371076584, 0.008706605061888695, 0.03288928046822548, 0.0012297948123887181, 0.016465695574879646, 0.03389996290206909, 0.03227865695953369, -0.009154042229056358, -0.02659357152879238, 0.015433957800269127, -0.0052323853597044945, -0.02465643174946308, -0.017392154783010483, 0.010201572440564632, 0.02225606143474579, 0.004379622172564268, 0.040658898651599884, 0.0014449595473706722, -0.016107745468616486, 0.0216454416513443, -0.004695460665971041, -0.010201572440564632, -0.04303821176290512, 0.009969957172870636, 0.0058798533864319324, -0.007364292629063129, 0.013402065262198448, 0.006369402632117271, 0.00022075770539231598, 0.009090874344110489, -0.043501440435647964, 0.020761094987392426, -0.02490910142660141, -0.009222473949193954, -0.015718212351202965, -0.003926921170204878, -0.034742195159196854, 0.014518027193844318, -0.03712151199579239, 0.018434420228004456, -0.0021147995721548796, -0.031289033591747284, -0.03579498827457428, -0.01109644677489996, -0.0006162133649922907, 0.036363497376441956, 0.02212972566485405, -0.015097063034772873, 0.01750796101987362, -0.018602866679430008, -0.026214566081762314, 0.005990396719425917, 0.002012152224779129, -0.023624693974852562, -0.004850747529417276, 0.014560138806700706, -0.0077590905129909515, -0.004861275665462017, -0.011686011217534542, -0.005271865054965019, 0.00859606172889471, -0.023182520642876625, 0.019181903451681137, -0.03331039845943451, 0.011106974445283413, 0.008838203735649586, -0.00903297122567892, -0.02236134000122547, 0.004221703391522169, 0.026151398196816444, 0.03293139114975929, -0.03389996290206909, 0.011359645053744316, 0.034426357597112656, 0.022866681218147278, -0.0010995115153491497, 0.013107283040881157, 0.0285938810557127, 0.026635684072971344, 0.031878598034381866, 0.027309471741318703, -0.010638481937348843, 0.014981255866587162, -0.01892923377454281, 0.0011251734104007483, 0.012128184549510479, -0.01139122899621725, 0.01867656223475933, -0.008459198288619518, -0.017002619802951813, -0.022171838209033012, 0.022592956200242043, -0.009890997782349586, -0.021245379000902176, -0.022508731111884117, 0.038426972925662994, 0.018002774566411972, 0.00780120212584734, 0.02064528688788414, -0.017823798581957817, 0.0033768361900001764, 0.028699159622192383, -0.013581040315330029, 0.005390304606407881, -0.005774574354290962, 0.033394619822502136, -0.03535281494259834, 0.015455013141036034, -0.018718674778938293, 0.018571283668279648, -0.03674250468611717, 0.0300467349588871, 0.009396185167133808, -0.02394053153693676, -0.004874435253441334, -0.02484593354165554, 0.02183494344353676, -0.007022134959697723, 0.010764816775918007, -0.015581348910927773, -0.0037742657586932182, -0.008364447392523289, 0.014275884255766869, -0.017760632559657097, -0.007616963237524033, -0.009838358499109745, 0.03149959072470665, 0.012759861536324024, -0.03728995844721794, 0.01667625457048416, -0.009996277280151844, 0.024593263864517212, -0.002464853459969163, -0.00038032178417779505, 0.007959120906889439, -0.027709534391760826, -0.028678104281425476, 0.020634759217500687, -0.022929849103093147, -0.012075545266270638, 0.027899036183953285, -0.02128749154508114, -0.06354663521051407, -0.038595423102378845, 0.011843930929899216, -0.035394929349422455, -0.007953857071697712, 0.018845008686184883, 0.008938219398260117, -0.00225824280641973, 0.0007086618570610881, -0.021624384447932243, -0.041143182665109634, 0.0102594755589962, 0.004563861060887575, 0.008585534058511257, 0.011264894157648087, -0.016813118010759354, 0.03244710713624954, 0.009959429502487183, 0.017076315358281136, -0.009043498896062374, 0.05701931193470955, 0.008890843950212002, 0.026404069736599922, 0.03507909178733826, 0.009785718284547329, -0.03882703557610512, -0.011991322040557861, -0.013096755370497704, 0.017276346683502197, -0.010559522546827793, 0.00019262837304268032, 0.01592877134680748, -0.013549456372857094, -0.03903759643435478, -0.022466620430350304, -0.04067995399236679, -0.026825185865163803, -0.004242759197950363, -0.0004010486591141671, 0.009506728500127792, -0.0052455454133450985, -0.0004543463292066008, -0.021445410326123238, -0.009717286564409733, 0.03387890383601189, -0.022340284660458565, 0.018181748688220978, -0.016907868906855583, 0.027098912745714188, -0.005124473944306374, -0.0007942013326101005, 0.037942688912153244, 0.01777116023004055, 0.006564169656485319, -0.010017333552241325, 0.020466312766075134, -0.0025767129845917225, -0.03453163802623749, -0.005324504803866148, 0.013886350207030773, 0.058619558811187744, -0.017939606681466103, 0.01570768468081951, -0.0014291675761342049, 0.05150267481803894, 0.005027090664952993, -0.0009080346790142357, 0.022887738421559334, 0.007780146319419146, -0.011328062042593956, -0.0022082349751144648, -0.02000308223068714, -0.03838486224412918, 0.03438424691557884, -0.009996277280151844, -0.022277116775512695, -0.016781533136963844, 0.007690658792853355, -0.01950826868414879, 0.001772641553543508, 0.018297556787729263, 0.013970574364066124, -0.014854921028017998, 0.005621918942779303, -0.009675174951553345, 0.0035821308847516775, 0.022719290107488632, 0.005406096111983061, 0.012507190927863121, 0.011959738098084927, 0.022592956200242043, 0.003079422051087022, -0.016455167904496193, 0.013317842036485672, -0.026804130524396896, 0.0015594508731737733, 0.015012839809060097, -0.01411796547472477, -0.005737726576626301, 0.006190427578985691, 0.00286359922029078, -0.06961072981357574, -0.01983463577926159, 0.02968878671526909, -0.014960200525820255, 0.022803515195846558, 0.009117194451391697, -0.010627953335642815, 0.016823645681142807, -0.018002774566411972, 0.022635066881775856, -0.005063938442617655, -0.0015147071098908782, -0.013465233147144318, 0.004358566366136074, -0.0016291984356939793, -0.005569279193878174, 0.025730280205607414, -0.022887738421559334, -0.026067174971103668, 0.02697257697582245, -0.003916393034160137, 0.01439169142395258, -0.020108362659811974, -0.0011613632086664438, 0.020729510113596916, 0.0052613369189202785, -0.005063938442617655, 0.007848577573895454, -0.014949671924114227, -0.01796066202223301, 0.004795475862920284, 0.008896107785403728, 0.022993016988039017, 0.03806902468204498, -0.019202958792448044, -0.032615553587675095, 0.013454705476760864, -0.03754262626171112, -0.011738651432096958, -0.03194176405668259, 0.02170860953629017, 0.010154196061193943, 0.04118529334664345, -0.014760169200599194, 0.04027989134192467, -0.009538312442600727, 0.027709534391760826, -0.01753954589366913, -0.046280816197395325, 0.010175252333283424, 0.03337356448173523, 0.008817148394882679, 0.014191661030054092, 0.023603636771440506, -0.005832477938383818, 0.04868118837475777, 0.02316146343946457, -0.0002011823235079646, -0.007674866821616888, 0.07554848492145538, -0.05125000327825546, 0.008590797893702984, 0.056008633226156235, -0.011833402328193188, 0.006795783992856741, 0.026509348303079605, 0.018329139798879623, 0.012612470425665379, 0.003971664700657129, 0.02545655518770218, 0.000927774584852159, -0.01870814710855484, -0.011233310215175152, 0.006885271519422531, 0.006027244497090578, 0.00823811162263155, 0.006516793742775917, -0.006079884245991707, 0.0033241964410990477, -0.024024754762649536, 0.006395722273737192, -0.018202805891633034, -0.027962204068899155, -0.02061370387673378, -0.008011761121451855, 0.029162388294935226, -0.037205733358860016, -0.029246613383293152, 0.019266126677393913, -0.01075955294072628, -0.0009994960855692625, -0.03918498754501343, 0.02646723762154579, -0.01948721334338188, -0.0036452985368669033, -0.008422350510954857, -0.002147699473425746, -0.032636608928442, 0.013254674151539803, -0.035521265119314194, 0.02112957276403904, -0.03120480850338936, -0.02823593094944954, -0.03131008893251419, 0.041985418647527695, 0.011243837885558605, 0.013138866983354092, 0.018845008686184883, -0.021476993337273598, -0.01577085070312023, -0.0017529017059132457, 0.002380630001425743, 0.040827345103025436, 0.015065480023622513, -0.0008172311936505139, 0.012507190927863121, 0.022277116775512695, -0.022087614983320236, 0.005821949802339077, -0.009080346673727036, 0.018729202449321747, -0.026340901851654053, 0.020276809111237526, 0.013317842036485672, -0.02083479054272175, 0.010654273442924023, -0.01780274324119091, -0.0429750457406044, 0.002425373764708638, 0.0093698650598526, -0.0033873640932142735, 0.021729664877057076, 0.006606281269341707, 0.0009955482091754675, 0.042659204453229904, -0.04927075281739235, -0.012970419600605965, -0.03179437294602394, 0.026635684072971344, 0.010270004160702229, 0.028214873746037483, 0.0029346628580242395, -0.010685857385396957, -0.017107900232076645, -0.049902427941560745, 0.02086637355387211, -0.004956027027219534, 0.006779992021620274, -0.007337972987443209, -0.010864832438528538, 0.02490910142660141, -0.0005454787751659751, -0.012359799817204475, 0.006616809405386448, -0.030509965494275093, 0.021498050540685654, 0.00011136584362247959, -0.02562500163912773, 0.028551768511533737, 0.040532562881708145, -0.01786591112613678, 0.002913606818765402, -0.031162697821855545, 0.0018200173508375883, -0.0005306738894432783, 0.019887275993824005, -0.017297402024269104, 0.029162388294935226, 0.03667933866381645, 0.0017186859622597694, -0.010575314052402973, 0.016813118010759354, 0.0060640922747552395, 0.015349733643233776, -0.017950134351849556, -0.004021672531962395, -0.001535763032734394, -0.005653502885252237, -0.026088230311870575, 0.0040690479800105095, 0.010780609212815762, -0.0054376800544559956, 0.009827830828726292, -0.012633525766432285, -0.01474964153021574, 0.01689734123647213, -0.016581503674387932, -0.023666804656386375, -0.011201726272702217, 0.006837896071374416, 0.0058851176872849464, 0.01906609535217285, -0.014665418304502964, -0.0015226030955091119, 0.033331453800201416, -0.05453471839427948, -0.03415263071656227, 0.04274342954158783, -0.03415263071656227, -0.020634759217500687, -0.02044525556266308, -0.008722396567463875, -0.030741579830646515, 0.021856000646948814, 0.0285938810557127, 0.010749025270342827, -0.002839911263436079, 0.015918241813778877, -0.004166431725025177, 0.03886914625763893, -0.004071679897606373, 0.009780454449355602, 0.0285938810557127, -0.025161772966384888, 0.011643899604678154, -0.007332709152251482, -0.013707376085221767, -0.04383833333849907, -0.03577393293380737, -0.04366988688707352, 0.007653811015188694, -0.0042401272803545, -0.008459198288619518, 0.03415263071656227, -0.03299455717206001, 0.025182828307151794, 0.02200339175760746, 0.009912054054439068, 0.024951213970780373, -0.01715001091361046, 0.001219924888573587, -0.003753209952265024, -0.010970111936330795, -0.0011784711387008429, 0.012886196374893188, -0.008622381836175919, 0.022698234766721725, 0.023014072328805923, -0.015118119306862354, 0.008780300617218018, -0.007480100262910128, -0.0016410424141213298, 0.01447591558098793, -0.0034531636629253626, 0.03383679315447807, 0.0840129479765892, 0.0062483311630785465, 0.0022714026272296906, -0.022824570536613464, 0.017044732347130775, -0.016939451918005943, 0.03499486669898033, 0.03307878226041794, -0.0016884180950000882, 0.021182211115956306, 0.007495892234146595, 0.0036374027840793133, 0.000816573214251548, 0.06485209614038467, -0.006169371772557497, -0.0040690479800105095, 0.01082272082567215, 0.030783692374825478, -0.02351941354572773, -0.008696077391505241, 0.009406712837517262, -0.02028733678162098, -0.016592031344771385, 0.014675945974886417, -0.015844546258449554, 0.009701495058834553, 0.004527013283222914, 0.021266436204314232, 0.0005451497854664922, 0.03670039400458336, -0.01499178446829319, 0.049776092171669006, 0.050197210162878036, -0.01806594245135784, 0.001233742805197835, 0.03320511803030968, -0.007938065566122532, -0.016265664249658585, 0.01586560346186161, 0.006290443241596222, -0.00909613911062479, -0.005340296775102615, -0.026677794754505157, 0.0026530404575169086, -0.028614936396479607, -0.011317533440887928, -0.025098605081439018, 0.020150473341345787, 0.014170604757964611, -0.0161603856831789, -0.012980948202311993, 0.03741629421710968, -0.0066010174341499805, 0.03579498827457428, -0.00011819255450973287, -0.0130020035430789, -0.021729664877057076, -0.00039084971649572253, 0.023140408098697662, 0.006311499048024416, -0.07217954844236374, -0.0005816685734316707, 0.008832939900457859, -0.007058982737362385, -0.0252038836479187]
```
</details>
