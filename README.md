# Qwen3-8B 気候変動ドメイン特化ファインチューニング

CONSEO気候変動レポートのPDF文書を使用して、Qwen3-8BモデルをQLoRA（Quantized Low-Rank Adaptation）でファインチューニングするプロジェクトです。

## 📋 プロジェクト概要

- **目的**: 気候変動に関する専門知識を持つローカルLLMの構築
- **ベースモデル**: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- **ファインチューニング手法**: QLoRA（4ビット/8ビット量子化）
- **データソース**: CONSEO気候変動レポート（PDF）
- **参考実装**: [book-local-llm-sample](https://github.com/kujirahand/book-local-llm-sample/tree/main/src/ch4/anime_llm)

## 🏗️ プロジェクト構造

```
fine-tuning-local-llm-for-conse/
├── documents/                    # 入力文書
│   ├── document_01.pdf          # CONSEO気候変動レポート（原本）
│   ├── extracted_cleaned.txt    # クリーンアップ済みテキスト
│   ├── extracted_pdfplumber.txt # pdfplumberによる抽出結果
│   ├── extracted_pymupdf_blocks.txt  # PyMuPDFブロック抽出結果
│   ├── extracted_pymupdf_direct.txt  # PyMuPDF直接抽出結果
│   └── extracted_pymupdf4llm.txt     # pymupdf4llm抽出結果
│
├── data/                         # 処理済みデータ
│   ├── corpus.txt               # コーパステキスト
│   ├── corpus_chunks.json       # チャンク分割されたコーパス
│   ├── dataset.json             # 基本データセット
│   ├── alpaca_dataset.json      # Alpaca形式データセット（2,243件）
│   ├── evaluation_results.json  # 4ビット版評価結果
│   └── evaluation_results_8bit.json  # 8ビット版評価結果
│
├── scripts/                      # 処理スクリプト
│   ├── 1-extract_pdf.py         # PDF抽出（基本版）
│   ├── 1-extract_pdf_compare.py # PDF抽出手法比較
│   ├── 1b-cleanup_extracted_text.py  # テキストクリーンアップ
│   ├── 2-prepare_corpus.py      # コーパス準備
│   ├── 3-make_dataset.py        # データセット生成
│   ├── 4-finetune_qwen3.py      # 4ビット版ファインチューニング
│   ├── 4-finetune_qwen3_8bit.py # 8ビット版ファインチューニング
│   ├── 5-test_model.py          # モデルテスト
│   ├── 6-evaluate_model.py      # 4ビット版評価
│   └── 6-evaluate_model_8bit.py # 8ビット版評価
│
├── models/                       # 出力モデル
│   ├── qwen3-8b-climate-lora/   # 4ビット版LoRAアダプター
│   ├── qwen3-8b-climate-lora-8bit/  # 8ビット版LoRAアダプター
│   └── qwen3-8b-climate-merged/ # マージ済みモデル（4ビット版）
│
├── logs/                         # 実行ログ
│   ├── finetune_final.log       # 4ビット版学習ログ
│   ├── finetune_8bit_unsloth_v4.log  # 8ビット版学習ログ
│   ├── evaluation.log           # 4ビット版評価ログ
│   └── evaluation_8bit_unsloth.log   # 8ビット版評価ログ
│
└── requirements.txt              # 依存パッケージ
```

## 🛠️ 環境構築

### 必要環境

- Python 3.10+
- CUDA 11.8+ 対応GPU（VRAM 24GB推奨）
- Conda

### セットアップ

```bash
# Conda環境の作成
conda create -n qwen3_training python=3.10 -y
conda activate qwen3_training

# PyTorchのインストール（CUDA 11.8）
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# 依存パッケージのインストール
pip install pymupdf4llm pymupdf pdfplumber
pip install transformers datasets peft accelerate bitsandbytes trl
pip install sentencepiece protobuf

# Unslothのインストール（4ビット/8ビット版ファインチューニング用）
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# 評価用パッケージ
pip install evaluate rouge_score bert_score sacrebleu
```

## 📖 実行手順

### Step 1: PDFからテキスト抽出

```bash
# 複数の抽出手法を比較
python scripts/1-extract_pdf_compare.py

# テキストのクリーンアップ（不要な改行を削除）
python scripts/1b-cleanup_extracted_text.py
```

### Step 2: コーパス準備

```bash
python scripts/2-prepare_corpus.py
```

### Step 3: データセット生成

```bash
python scripts/3-make_dataset.py
```

生成されるデータセット: **2,243件**のQ&Aペア（データ拡張により増幅、図表参照は除外）

### Step 4: ファインチューニング

```bash
# 4ビット版（推奨）
python scripts/4-finetune_qwen3.py

# 8ビット版（環境変数設定が必要）
UNSLOTH_DISABLE_TORCH_COMPILE=1 TORCH_COMPILE_DISABLE=1 python scripts/4-finetune_qwen3_8bit.py
```

### Step 5: モデルテスト

```bash
python scripts/5-test_model.py
```

### Step 6: 評価

```bash
# 4ビット版の評価
python scripts/6-evaluate_model.py

# 8ビット版の評価
UNSLOTH_DISABLE_TORCH_COMPILE=1 TORCH_COMPILE_DISABLE=1 python scripts/6-evaluate_model_8bit.py
```

## 📊 評価結果

### 4ビット版 vs 8ビット版 比較

両バージョンともに優れた性能を達成しました。

| 指標 | ベースモデル | 4ビット Fine-tuned | 8ビット Fine-tuned |
|------|-------------|-------------------|-------------------|
| **BLEU** | 0.0% | 91.93% ✅ | 91.93% ✅ |
| **BLEU-1** | 0.0% | 92.86% | 92.86% |
| **BLEU-2** | 0.0% | 92.31% | 92.31% |
| **BLEU-3** | 0.0% | 91.67% | 91.67% |
| **BLEU-4** | 0.0% | 90.91% | 90.91% |
| **ROUGE-1** | 11.1-13.9% | 39.14% | 37.12% |
| **ROUGE-2** | 2.3-2.7% | 33.74% | 31.83% |
| **ROUGE-L** | 10.8-13.5% | 39.14% | 36.88% |
| **BERTScore Precision** | 63.8-64.7% | 71.52% | 70.81% |
| **BERTScore Recall** | 62.8-63.0% | 85.58% | 85.44% |
| **BERTScore F1** | 63.2-63.5% | 77.41% ✅ | 76.94% ✅ |

### 学習パラメータ比較

| パラメータ | 4ビット版 | 8ビット版 |
|-----------|----------|----------|
| ベースモデル | unsloth/Qwen3-8B-bnb-4bit | unsloth/Qwen3-8B |
| 量子化 | load_in_4bit=True | load_in_8bit=True |
| LoRA rank (r) | 16 | 16 |
| LoRA alpha | 16 | 16 |
| LoRA dropout | 0 | 0 |
| 学習率 | 2e-4 | 2e-4 |
| バッチサイズ | 2 | 1 |
| 勾配蓄積ステップ | 4 | 8 |
| エポック数 | 3 | 3 |
| 最大シーケンス長 | 2048 | 2048 |
| 学習時間 | 約100分 | 約69分 |
| GPU メモリ使用量 | 約8GB | 約10GB |
| Gradient Checkpointing | "unsloth" | False（無効化が必要） |

### 📝 学習前後のモデル回答例（4ビット版）

以下は、ベースモデル（学習前）とFine-tunedモデル（学習後）の回答を比較した例です。

---

#### 例1: 気候変動と国際協力について

**質問**: コラム（その3） 気候変動と国際協力 （角南篤CONSEO について説明してください。

**ベースモデル（学習前）**:
> 角南篤（すみだ あきら）は、日本の国際問題研究者で、CONSEO（国際協力と環境戦略研究センター）の代表理事を務めています。彼は、気候変動と国際協力に関する研究を専門としており...

**Fine-tunedモデル（学習後）** ✅:
> ご質問にお答えします。コラム（その3） 気候変動と国際協力 （角南篤CONSEO 会長）
> 私たちの住む地球は、気候変動の影響が如実に感じられる事象が世界中で多発しているだけでなく、最近では生物多様性や感染症の課題など、問題が複雑化しています。気候変動の理解は、一国の科学技術ではなし得ず、その対応も然りです。
> そこでCONSEO では、気候変動の理解と対応は世界の協力で成り立っている点について今一度焦点を当てることとしました。気候変動シリーズで第1 回（7 月19 日）・第5 回（11 月20 日）と二回にわたり【気候変動の理解は、世界の協力で成り立っている！】を開催し...

**評価**: ベースモデルは名前の読み方を誤り一般的な説明。Fine-tunedモデルはCONSEOの具体的な活動内容を正確に回答。

---

#### 例2: CSA（カナダ宇宙庁）について

**質問**: CSAについて解説してください。

**ベースモデル（学習前）** ❌:
> CSA（Consumer Supported Agriculture）は、消費者が農業生産者と直接つながり、農産物を購入することで、農業の持続可能な発展を支援する制度です...

**Fine-tunedモデル（学習後）** ✅:
> ◼ SWOT2022 年12 月、NASA とフランス国立宇宙研究センター（CNES）が主導し、カナダ宇宙庁（CSA）と英国宇宙庁（UKSA）が参画するSWOT 衛星が打ち上げられました...

**評価**: ベースモデルは農業用語と誤解。Fine-tunedモデルは宇宙開発の文脈でカナダ宇宙庁として正しく認識。

---

#### 例3: 雪氷面のアルベドについて

**質問**: たとえば雪氷面についてわかりやすく説明してください。

**ベースモデル（学習前）**:
> 雪氷面とは、雪や氷が積もった地面のことです。冬になると、雪が降って地面が白くなります...雪氷面は、冬の自然の一部で、雪や氷が積もることで、自然の美しさや、冬の風物詩になります。

**Fine-tunedモデル（学習後）** ✅:
> たとえば雪氷面は、そのアルベド（太陽放射の波長全域に対する反射率）が大きいのに対して雪氷がない陸面や海面のアルベドは小さいため、温暖化により雪氷が融けると地球はより太陽放射エネルギーを吸収しやすくなります。

**評価**: ベースモデルは一般的で非科学的な説明。Fine-tunedモデルはアルベドという専門用語を用いて気候変動の観点から科学的に説明。

---

#### 例4: 地球デジタルツインエンジンについて

**質問**: 地球デジタルツインエンジンとは何でしょうか？

**ベースモデル（学習前）**:
> The Earth Digital Twin Engine is a system that creates a digital twin of the Earth, which is a virtual model of the real Earth...

**Fine-tunedモデル（学習後）** ✅:
> A) 地球デジタルツインの構成地球デジタルツインを実現するためには、「地球デジタルツインエンジン」「データレイク」「ユーザーインターフェース」の3 つの要素が必要です。「地球デジタルツインエンジン」は、地球デジタルツインの中核部分かつ、地球デジタルツイン上のデータを統合・再構成させるための技術であり、各種観測データの処理や解析、シミュレーション・予測を行います。

**評価**: ベースモデルは英語で回答を開始し一般的な説明。Fine-tunedモデルは日本語で3つの構成要素を含む正確な説明。

---

### 📝 学習前後のモデル回答例（8ビット版）

8ビット版も同様に優れた結果を示しています。

---

#### 例1: CSA（カナダ宇宙庁）について

**質問**: CSAについて解説してください。

**ベースモデル（8ビット）** ❌:
> CSA（Consumer Supported Agriculture）は、消費者が農業生産者と直接つながり、農産物を購入することで、農業の持続可能な発展を支援する制度です。この制度は、地元の農業者と消費者が直接つながり...

**Fine-tunedモデル（8ビット）** ✅:
> ◼ SWOT2022 年12 月、NASA とフランス国立宇宙研究センター（CNES）が主導し、カナダ宇宙庁（CSA）と英国宇宙庁（UKSA）が参画するSWOT 衛星が打ち上げられました。さらに、◼ 気候変動への予測と対応 近年の気候変動に係る予測では...

---

#### 例2: 雪氷面について

**質問**: たとえば雪氷面についてわかりやすく説明してください。

**ベースモデル（8ビット）**:
> 雪氷面とは、雪や氷が積もった地面のことです。冬になると、雪が降って地面が白くなります。その雪が融けて、水になって川や湖に流れていきます...

**Fine-tunedモデル（8ビット）** ✅:
> たとえば雪氷面は、そのアルベド（太陽放射の波長全域に対する反射率）が大きいのに対して雪氷がない陸面や海面のアルベドは小さいため、温暖化により雪氷が融けると地球はより太陽放射エネルギーを吸収しやすくなります。

---

これらの例から、**4ビット版と8ビット版の両方**で以下の改善が確認されました：

1. **ドメイン特化知識**: 気候変動・地球観測分野の専門用語・概念を正確に理解
2. **文脈に応じた回答**: 同じ用語（CSA、NASA等）でも気候変動の文脈で適切に回答
3. **具体的な情報**: 図表番号、出所、具体的な数値やプログラム名を含む回答
4. **日本語の一貫性**: 英語での思考を開始せず、日本語で一貫した回答

## 🔧 試行錯誤と解決策

### 1. PDFテキスト抽出の問題

**問題**: 初期の`pymupdf4llm`による抽出では、以下の問題が発生
- 過剰な改行（`\n\n`）
- 2カラムレイアウトによる文章の分断
- 図表キャプションの混入

**解決策**:
1. 複数の抽出手法を比較（`pymupdf4llm`, `pdfplumber`, `PyMuPDF`直接, `PyMuPDF`ブロック）
2. `PyMuPDF`ブロック抽出を採用（2カラム対応が優秀）
3. 句読点（`。`）に基づく文結合処理を実装
4. ページヘッダー/フッターの除去

```python
# 1b-cleanup_extracted_text.py の主要ロジック
def combine_sentences(text):
    """句点で終わっていない行を次の行と結合"""
    lines = text.split('\n')
    combined = []
    buffer = ""
    for line in lines:
        if buffer:
            buffer += line
        else:
            buffer = line
        if buffer.endswith('。') or buffer.endswith('）') or not buffer:
            combined.append(buffer)
            buffer = ""
    return '\n'.join(combined)
```

### 2. CUDA Out of Memory エラー

**問題**: Qwen3-8Bモデル（約16GB）のロードでOOMエラー

**解決策**:
1. `unsloth`ライブラリの4ビット量子化版を使用
2. モデル名を`unsloth/Qwen3-8B-bnb-4bit`に変更
3. 勾配チェックポイントの有効化

```python
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B-bnb-4bit",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### 3. SFTTrainer の引数エラー

**問題**: `SFTTrainer.__init__() got an unexpected keyword argument 'tokenizer'`

**解決策**: `unsloth`版の`SFTTrainer`では`tokenizer`引数が不要

```python
# 修正前
trainer = SFTTrainer(model=model, tokenizer=tokenizer, ...)

# 修正後（processing_classを使用）
trainer = SFTTrainer(model=model, processing_class=tokenizer, ...)
```

### 4. 学習プロセスの中断

**問題**: 長時間の学習中にプロセスが予期せず停止

**解決策**: `nohup`を使用してバックグラウンド実行

```bash
nohup python -u scripts/4-finetune_qwen3.py 2>&1 | tee logs/finetune.log &
```

### 5. 8ビット版 QLoRA の問題と解決 ✅

8ビット版のファインチューニングでは、複数の問題に直面しましたが、最終的に解決しました。

#### 問題1: transformers + peft による8ビット量子化の失敗

**症状**:
- `grad_norm: 0.0`が全ステップで発生
- Lossが全く減少しない（約2.75で横ばい）
- 推論時にLoRAアダプターが効かない

**原因**: `bitsandbytes`の8ビット量子化（`MatMul8bitLt`関数）が勾配計算をサポートしない

**試行した解決策**（すべて失敗）:
```python
# 1. 勾配チェックポイントの無効化
prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)

# 2. LoRAパラメータをfloat32に変換
for name, param in model.named_parameters():
    if "lora_" in name and param.requires_grad:
        param.data = param.data.to(torch.float32)

# 3. オプティマイザの変更
optim="adamw_torch"  # adamw_8bitの代わりに
```

#### 問題2: Unsloth 8ビット + torch.compile の互換性エラー

**症状**: 学習開始直後に `AssertionError: wrong number of dimensions` エラー

**原因**: Unslothの内部コンパイルキャッシュと8ビット量子化の互換性問題

#### 最終解決策 ✅

**Unslothベースのスクリプトに書き換え + 環境変数設定**:

```python
# scripts/4-finetune_qwen3_8bit.py
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen3-8B",
    max_seq_length=2048,
    load_in_4bit=False,
    load_in_8bit=True,  # 8ビット量子化を有効化
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=False,  # 8ビット互換性のため無効化
    random_state=42,
)
```

**実行時の環境変数設定**:
```bash
UNSLOTH_DISABLE_TORCH_COMPILE=1 TORCH_COMPILE_DISABLE=1 python scripts/4-finetune_qwen3_8bit.py
```

**結果**:
- Loss: 2.4168 → 0.0527 に減少 ✅
- grad_norm: 正常に更新 ✅
- 評価結果: 4ビット版とほぼ同等の性能 ✅

### 6. 図表参照の除外

**問題**: データセットに「図表+数字」（例: 図表 46）を含むエントリがあり、学習に不適切

**解決策**: `scripts/3-make_dataset.py`にフィルタリング処理を追加

```python
def filter_figure_references(qa_pairs):
    """質問または回答に「図表+数字」のパターンを含むQ&Aペアを除外"""
    figure_pattern = re.compile(r"図表\s*\d+")
    return [qa for qa in qa_pairs 
            if not (figure_pattern.search(qa["instruction"]) or 
                    figure_pattern.search(qa["output"]))]
```

結果: 4,148件 → 2,243件（1,905件を除外）

## 📈 データセット生成の工夫

参考実装に基づき、以下のデータ拡張を実装:

1. **複数の質問テンプレート**: 同じ内容に対して異なる質問形式を生成
2. **言い換え**: 「ですます調」↔「である調」の変換
3. **丁寧度の変更**: フォーマル/カジュアルな表現の追加
4. **組み合わせQ&A**: 複数のトピックを組み合わせた質問
5. **図表参照の除外**: 「図表+数字」パターンを含むエントリを除外（学習に不適切なため）

```python
QUESTION_TEMPLATES = [
    "{topic}について教えてください。",
    "{topic}とは何ですか？",
    "{topic}について詳しく説明してください。",
    "{topic}の概要を教えてください。",
    "{topic}について解説してください。",
]
```

結果: 基本データ約400件 → **2,243件**に拡張（図表参照1,905件を除外後）

## 🎯 結論

### 成功点

1. **4ビット版・8ビット版両方でQLoRAによる効率的なファインチューニング**
   - 4ビット版: 24GB GPUで約100分で学習完了、メモリ約8GB使用
   - 8ビット版: 24GB GPUで約69分で学習完了、メモリ約10GB使用
   - 両バージョンで BLEU 91.93%、BERTScore F1 77%前後の大幅改善

2. **日本語PDFからの効果的なデータセット生成**
   - 複数の抽出手法を比較・選定
   - 適切な前処理パイプラインの構築
   - データ拡張による学習データの増幅

3. **ドメイン特化型回答の実現**
   - 気候変動に関する専門的な質問に対して、学習データに基づいた正確な回答を生成

4. **8ビット量子化の問題解決**
   - Unslothベースのスクリプトと環境変数設定により、8ビット版も正常動作

### 4ビット版 vs 8ビット版の選択指針

| 観点 | 4ビット版 | 8ビット版 |
|-----|----------|----------|
| **メモリ効率** | ◎ (約8GB) | ○ (約10GB) |
| **学習速度** | ○ (約100分) | ◎ (約69分) |
| **セットアップの容易さ** | ◎ (標準設定) | △ (環境変数設定必要) |
| **評価性能** | ◎ (BERTScore F1: 77.41%) | ◎ (BERTScore F1: 76.94%) |
| **推奨度** | ★★★★★ | ★★★★☆ |

**推奨**: 特別な理由がなければ **4ビット版を使用** してください。セットアップが容易で、性能も同等以上です。

### 今後の課題

1. **評価データセットの分離**: 学習データと評価データの完全分離
2. **より大規模なデータセット**: 複数のPDF文書からのデータ収集
3. **ハイパーパラメータチューニング**: 学習率、LoRA rank等の最適化
4. **Flash Attention 2の導入**: さらなる高速化・メモリ効率化

## 📚 参考資料

- [Unsloth GitHub](https://github.com/unslothai/unsloth)
- [Qwen3-8B on Hugging Face](https://huggingface.co/Qwen/Qwen3-8B)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [book-local-llm-sample](https://github.com/kujirahand/book-local-llm-sample)

## 📄 ライセンス

本プロジェクトのコードはMITライセンスで公開されています。
学習に使用したCONSEO気候変動レポートの著作権は原著作者に帰属します。
