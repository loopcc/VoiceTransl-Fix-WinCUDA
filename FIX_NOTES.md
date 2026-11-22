# VoiceTransl 修复版 — Sakura + Qwen3 空译文修复

> 本仓库是 [shinnpuru/VoiceTransl](https://github.com/shinnpuru/VoiceTransl) 的**修改分支**。
> **只提供 Windows CUDA 版**，**不附带任何模型权重**（下载地址见下方[模型](#模型下载与保存路径)）。
> 原项目为 GPL-3.0，本分支作为派生作品遵循同一许可证。

---

## 为什么单独开这个分支

上游修复该 bug 的节奏比较慢，而这个问题不像一般的"翻译得不好"——它会**整句整句地丢译文**，成片字幕大段变成日文原文，没法等。

所以从源码定位、修复、重新打包，单独发布一版。改动只有 14 行，**欢迎上游直接合并**，合并后本分支即完成使命。

---

## 一、修复了什么

### 总览：修复清单

| # | 问题 | 表现 | 状态 |
| --- | --- | --- | --- |
| 1 | Sakura Qwen3 输出空译文 | 约 13% 句子变成 `(Failed)原文` | ✅ 已修 |
| 2 | 听写被切成 3 字碎片 | 字幕变成「先輩気 / 持ちい / い」，翻译全乱、速度极慢、CrispASR 反复报行数不一致 | ✅ 已修 |
| 3 | VAD 模型缺失 | 每次运行都联网下载 `firered-vad.gguf`，失败时 CrispASR 直接退出（code 1） | ✅ 已内置 |
| 4 | 原始字幕无法选择不留存 | 听写出的日文 `.srt` 一直堆在视频旁边 | ✅ 新增开关 |

> 第 2 项是「升级后才出现」的坑：`听写句子长度` 滑条在 v1.30 从不传给 CrispASR，历史保存值等于摆设；
> v1.31 起才真正生效，于是旧配置里的 `3` 被换算成 `--max-len 9`（3 个日文字），把字幕剁成碎片。
>
> 第 3 项导致 `[ERROR] CrispASR exited with code 1`：`crispasr/cache/` 里没有 VAD 模型时，CrispASR 会先尝试
> 从 HuggingFace 下载，网络不通就直接失败。本分支的发行包已内置该模型。

### 1.1 Sakura Qwen3 输出空译文

#### 现象

用本地 Sakura 模型（Qwen3 版）翻译日文字幕时，**约 13% 的句子**最后变成 `(Failed)原文`：

```
1
00:00:04,320 --> 00:00:06,160
(Failed)すっごい雨でしたね
```

一个 209 句的样本里挂掉 27 句。日志里堆满两类告警（共 302 次）：

```
[WARNING]...json:1~16]错误的输出：翻译结果行数 与 原文行数 不一致
[WARNING]...json:1]错误的输出：第1句空白
[ERROR]  ...json:1]单句循环重试5次出错，填充原文
```

翻不出来的句子有明显特征——**几乎全是极短句和拟声词**：`ん`、`あ`、`くっ`、`はっ`、`びしょびしょ`、`お疲れ様でした。`

#### 根因

`llama.cpp` 的日志暴露了真相：

```
prompt eval time = 33.72 ms / 1 tokens
      eval time =  0.00 ms / 1 tokens     ← 只生成 1 个 token 就停了
```

**模型第一个 token 就是 EOS**，也就是"什么都不想说"，返回空字符串。程序判定为空 → 重试 5 次 → 每次都是空 → 填回原文。

为什么会这样？链条是这样的：

1. `GalTransl/Thinking.py` 的 `thinking_body()` 对**本地 loopback 接口无条件**注入
   `chat_template_kwargs = {"enable_thinking": false}`（因为 GUI 的"思考模式"开关默认关闭）。
2. Qwen3 系的 chat template 在 `enable_thinking=false` 时，会在 assistant 头后面**预填一个空的 think 块**：

   ```jinja
   {{- '<|im_start|>assistant\n' }}
   {%- if enable_thinking is defined and enable_thinking is false %}
       {{- '<think>\n\n</think>\n\n' }}
   {%- endif %}
   ```

3. **Sakura 14B Qwen3 v1.5 是"直接出译文、不写思考块"的微调模型**。被塞了一个空的 `<think></think>` 之后，它认为"思考已经结束、没什么要输出"，于是立刻 EOS。

**为什么老版本没事**：以前用的 Sakura 是 Qwen2.5 系（`Sakura-14B-Qwen2.5-v1.0`），基座根本没有 thinking 这个概念，模板里也没有这个分支，`enable_thinking` 参数是空转的。换到 Qwen3 系之后才踩中。

#### 证据

同一批 16 句，同一个模型，只改这一个开关：

| 请求 | 结果 |
| --- | --- |
| `enable_thinking=false`（修复前） | 生成 **1 token**，返回 **0 行** ❌ |
| 不注入该参数（修复后） | 生成 **76 token**，返回 **16 行，全部正确** ✅ |

抓包也确认真实请求体里确实带着这个参数：

```json
"temperature":0.1,"top_p":0.8,"chat_template_kwargs":{"enable_thinking":false}
```

#### 代码改动

`GalTransl/Thinking.py`，只动 loopback 分支：

```python
NON_REASONING_MODEL_HINTS = ('sakura', 'galtransl', 'honyaku', 'translate-gemma')


def is_non_reasoning_finetune(model):
    name = str(model or '').lower()
    return any(hint in name for hint in NON_REASONING_MODEL_HINTS)


...
    if is_loopback_endpoint(endpoint):
        if not enabled and is_non_reasoning_finetune(model):
            return {}      # 翻译专模不注入该参数，交给服务端原生模板
        return {'chat_template_kwargs': {'enable_thinking': enabled}}
```

`tests/test_thinking_requests.py` 增加回归测试 `test_loopback_translation_finetune_skips_thinking_switch`。

#### 影响面

| 场景 | 行为 |
| --- | --- |
| 本地 Sakura / GalTransl 等翻译专模 + 关思考 | **不再注入参数**（修的就是这个）✅ |
| 本地 Sakura 等 + 开思考 | 不变，照旧注入 `true` |
| 本地 qwen3 等通用推理模型 + 关思考 | 不变，照旧注入 `false` |
| 远程 DeepSeek / 月之暗面 / 智谱 / 火山 / 硅基流动 / 阿里云 | 一行未动 |
| Ollama | 一行未动 |

只按**模型名**判断，不改变任何上游协议分支。

#### 验证

| 项目 | 结果 |
| --- | --- |
| 209 句完整样本（修复前） | 27 句 `(Failed)` ❌ |
| 209 句完整样本（修复后） | **0 句失败** ✅ |
| 单测 `tests/test_thinking_requests.py` | 3/3 通过 |
| 其它测试文件 | 通过（`test_headless_skill` 的报错是 PowerShell 输出编码问题，与本次改动无关，上游同样存在） |

修复后连原来完全翻不出来的都正常了：

```
ん              → 嗯
あ              → 啊
くっ            → 咕
はっ            → 哈
びしょびしょ      → 湿答答的
お疲れ様でした。   → 辛苦了。
```

---

### 1.2 听写被切成 3 字碎片（升级后「变慢、结果不对」的真凶）

#### 现象

升级后翻译变得极慢，译文也乱七八糟。日志里上千条：

```
[WARNING]...json:626~629]错误的输出：翻译结果行数 与 原文行数 不一致
```

同一个 41 分钟的视频，听写结果从 **209 段暴涨到 639 段**，字幕长这样：

```
1
00:00:42,070 --> 00:00:42,950
先輩気

2
00:00:43,030 --> 00:00:43,350
持ちい

3
00:00:43,350 --> 00:00:43,510
い

4
00:00:42,070 --> 00:00:43,990
ですか
```

「先輩気持ちいいですか」被劈成四段。翻译模型拿到这种碎片必然合并、漏行，
于是触发大量重试，既慢又错。**问题出在听写，不在翻译。**

#### 根因

`app.py` 的 `_build_crispasr_command()` 会按「听写句子长度」滑条拼 `--max-len`：

```python
bytes_per_char = 3 if language in ('ja', 'zh', 'ko') else ...
command = filtered + ['--max-len', str(int(max_chars) * bytes_per_char)]
```

日文按 UTF-8 每字 3 字节换算。配置里的 `asr_max_chars: 3` → `--max-len 9` → **每句最多 3 个字**。

这个滑条 **v1.30 及更早版本从不传给 CrispASR**（老日志里根本没有 `--max-len`），
所以历史保存的 `3` 一直是无害的摆设；v1.31 起它才真正生效，一升级就爆。

#### 实测

同一段 3 分钟音频，只改这一个参数：

| 参数 | 段数 | 输出样例 |
| --- | --- | --- |
| `--max-len 9` | 42 段 | `すっご` / `い固く` / `なった` ❌ |
| 不加 `--max-len` | 10 段 | `すっごい固くなった` ✅ |

#### 改动

加载配置时把 1~5 这种「旧版遗留、语义上不可能是有意设置」的值归零，恢复「沿用参数文件」：

```python
saved_max_chars = int(gui_settings.get('asr_max_chars', 0) or 0)
if 0 < saved_max_chars <= 5:
    saved_max_chars = 0
self.asr_length_slider.setValue(saved_max_chars)
```

> 建议：这个滑条保持 **0（沿用参数文件）**。真要限制，也别低于 20 字，否则会切断句子。

### 1.3 VAD 模型缺失导致 CrispASR 直接退出

#### 现象

第二个视频直接失败：

```
[crispasr] crispasar[vad]: downloading https://huggingface.co/cstr/firered-vad-GGUF/resolve/main/firered-vad.gguf
[crispasr] crispasr: WinHTTP failed, falling back...
[ERROR] CrispASR exited with code 1
```

#### 根因

`_internal/crispasr/cache/` 下没有 `firered-vad.gguf`。CrispASR 启用 `--vad --vad-model firered` 时会先尝试下载，
本机网络不通就直接退出。

#### 改动

发行包已内置 `crispasr/cache/firered-vad.gguf`（2.3 MB，来自
[cstr/firered-vad-GGUF](https://huggingface.co/cstr/firered-vad-GGUF)），开箱即用，不再联网。

### 1.4 新增：翻译完成后不保存原始字幕文件

听写会把日文 `.srt` 写在视频旁边，翻译完其实没用，但会越堆越多。现在给了一个开关：

- 位置：**工作台 → 字幕输出**，`文件格式` 下方
- 默认：**已勾选**（不保存）
- 勾选时：译文 `.tg.srt` 写好后删除听写产出的日文 `.srt`
- 安全策略：**你自己提供的 SRT 输入永远不会被删除**；视频旁边已存在的同名 SRT 也视为你的文件，保留

三种情况实测：

| 场景 | 原文 `.srt` | 译文 `.tg.srt` |
| --- | --- | --- |
| 默认（勾选不保存） | 删除 | 保留 |
| 取消勾选 | 保留 | 保留 |
| 勾选，但 SRT 是用户提供的 | 保留 | 保留 |

## 二、下载

到 [Releases](../../releases) 下载 **`VoiceTransl-Fix-WinCUDA.zip`**（约 1.8 GiB）。

- 内置 `llama.cpp b10430` CUDA 12.4 版、`CrispASR v0.8.28` CUDA 版、`ffmpeg 9.0.1`
- **不含模型**，首次使用请按下一节放好权重
- SHA256：`bbfeb2276013392d0d66035d7e92c96d35d95679996b1cc1b156fdb54a45d81d`
- 已内置 `crispasr/cache/firered-vad.gguf`，不需要联网下载

解压后目录结构：

```
VoiceTransl/
├─ VoiceTransl.exe
└─ _internal/
   ├─ llama/        ← 翻译模型放这里
   ├─ crispasr/     ← 语音识别模型放这里
   ├─ ffmpeg/
   ├─ translate/
   └─ separate/
```

---

## 三、模型下载与保存路径

> 翻译模型主要来自 [**SakuraLLM**](https://huggingface.co/SakuraLLM)（共 48 个仓库，含 14B/8B/7B/1.5B 各版本），
> 其余 ASR 模型在 HuggingFace 上也都有。

### 3.1 翻译模型 → 放到 `_internal/llama/`

| 文件名 | 大小 | 说明 | 下载 |
| --- | --- | --- | --- |
| `sakura-14b-qwen3-v1.5-q6k.gguf` | 11.3 GB | **推荐**，Q6_K 质量最好，16GB 显存可跑 | [SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen3-v1.5-GGUF) |
| `sakura-14b-qwen3-v1.5-iq4xs.gguf` | 7.6 GB | 显存紧张时用（12GB 卡） | 同上 |
| `sakura-14b-qwen2.5-v1.0-q6k.gguf` | ~11 GB | 旧 Qwen2.5 版，想对比时用 | [SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF](https://huggingface.co/SakuraLLM/Sakura-14B-Qwen2.5-v1.0-GGUF) |

其它可选（同在 [huggingface.co/SakuraLLM](https://huggingface.co/SakuraLLM)）：

- `Sakura-GalTransl-14B-v3.8` / `Sakura-GalTransl-7B-v3.7` — Galgame 风格
- `Sakura-8B-Qwen3-v1.1-SFT-GGUF` — 更小更快
- `Sakura-1.5B-Qwen2.5-v1.0-GGUF` — 极低配兜底

> 注：本修复针对的是 **Qwen3 系** Sakura。如果你的模型名里不含 `sakura`，判断逻辑不会生效，
> 请把权重文件保持原名（或改 `GalTransl/Thinking.py` 里的 `NON_REASONING_MODEL_HINTS`）。

### 3.2 语音识别模型 → 放到 `_internal/crispasr/`

| 文件名 | 大小 | 用途 | 来源 |
| --- | --- | --- | --- |
| `whisper-large-v3-turbo-F16.gguf` | 1.5 GB | 默认 ASR（whisper） | [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) 系权重，CrispASR 可直接加载 |
| `qwen3-asr-1.7b-ja-anime-q8_0.gguf` | 2.3 GB | **日语动漫/AV 专精**，识别质量最好 | [cstr/qwen3-asr-1.7b-ja-anime-GGUF](https://huggingface.co/cstr/qwen3-asr-1.7b-ja-anime-GGUF)（基座 [jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf](https://huggingface.co/jaykwok/Qwen3-ASR-1.7B-JA-Anime-Galgame-hf)） |
| `qwen3-asr-1.7b-ja-anime-q4_k.gguf` | 1.4 GB | 同上，量化版 | 同上 |
| `qwen3-asr-1.7b-q4_k.gguf` | 1.2 GB | 通用 Qwen3-ASR | [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) |
| `qwen3-forced-aligner-0.6b-f16.gguf` | 1.7 GB | 强制对齐（出精确时间轴，**推荐**） | [Qwen/Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) |
| `qwen3-forced-aligner-0.6b-q4_k.gguf` | 0.5 GB | 同上，量化版 | 同上 |

### 3.3 VAD 模型 → `_internal/crispasr/cache/`

| 文件名 | 用途 | 说明 |
| --- | --- | --- |
| `firered-vad.gguf` | VAD 语音活动检测 | **本发行包已内置，无需下载**。自己从源码构建时需要手动放一份：[cstr/firered-vad-GGUF](https://huggingface.co/cstr/firered-vad-GGUF)，否则 CrispASR 会尝试联网下载，失败即退出 |

### 3.4 模型选择建议

- 翻译：`sakura-14b-qwen3-v1.5-q6k.gguf`（16GB 显存很稳）
- ASR：日文番剧/AV 用 `qwen3-asr-1.7b-ja-anime-q8_0.gguf` + `qwen3-forced-aligner-0.6b-f16.gguf`
- ASR 与翻译不会同时驻留显存，可以都用大个的

---

## 四、安装 / 从旧版升级

### 全新安装
1. 解压 `VoiceTransl-win-cuda-fix.zip`
2. 按第三节把模型放进对应目录
3. 双击 `VoiceTransl.exe`

### 从旧版升级（保留已下好的模型）
模型有几十 G，不用重下。两种方式任选：

**方式 A（推荐）**：把旧目录里的 `*.gguf` 硬链接/复制进新目录
```powershell
# 同盘用硬链接，瞬间完成且不占空间
$old = 'E:\Program\VoiceTransl\_internal'
$new = '<新解压目录>\_internal'
foreach ($d in 'llama','crispasr') {
  Get-ChildItem "$old\$d" -Filter *.gguf | ForEach-Object {
    New-Item -ItemType HardLink -Path "$new\$d\$($_.Name)" -Target $_.FullName -Force
  }
}
```

**方式 B**：直接用新目录覆盖旧目录，但**保留**这些东西：
- `_internal/llama/*.gguf`、`_internal/crispasr/*.gguf`（模型）
- `_internal/gui_settings.yaml`（你的设置）
- `_internal/.env`（API Key）
- `_internal/project/`（字典、缓存）

---

## 五、自己构建

```bash
# Python 3.11
pip install -r requirements.txt

# 运行时二进制：从官方 release 取 llama.cpp / CrispASR / ffmpeg 放进 llama/ crispasr/ ffmpeg/
#   llama.cpp  b10430  cuda-12.4：https://github.com/ggml-org/llama.cpp/releases
#   CrispASR   v0.8.28 cuda      ：https://github.com/CrispStrobe/CrispASR/releases
#   ffmpeg     9.0.1             ：https://github.com/GyanD/codexffmpeg/releases

python -m PyInstaller --clean --noconfirm translate.spec --distpath .
python -m PyInstaller --clean --noconfirm separate.spec  --distpath .
python -m PyInstaller --clean --noconfirm app.spec
# 产物：dist/VoiceTransl/
```

---

## 六、同步上游

```bash
git remote add upstream https://github.com/shinnpuru/VoiceTransl.git
git fetch upstream
git merge upstream/main
```

改动集中在 `GalTransl/Thinking.py` 和 `tests/test_thinking_requests.py` 两个文件，冲突面很小。

---

## 七、许可与致谢

- 原项目：[shinnpuru/VoiceTransl](https://github.com/shinnpuru/VoiceTransl)，GPL-3.0，本分支同样以 GPL-3.0 发布
- 翻译模型：[SakuraLLM](https://huggingface.co/SakuraLLM)，CC BY-NC-SA 4.0（**非商用**）
- 语音识别：[CrispASR](https://github.com/CrispStrobe/CrispASR)、[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B)
- 推理后端：[llama.cpp](https://github.com/ggml-org/llama.cpp)
