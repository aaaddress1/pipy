# pipy

pipy 是用 Python 移植的 Oh My Pi runtime。從 `mini_pi.py` 的 `run_agent()` 可以看到主迴圈。它呼叫模型、派送工具批次，再用每個呼叫的 ID 按順序把結果放回對話。模型回傳最後答覆時結束當輪。Session 與壓縮放在 `runtime.py`，自動記憶放在 `memory.py`。

上游基準是 Oh My Pi commit `24439066040e006eb08da7b204f0ac41c79d2de9`，上游 MIT 授權保存在 `LICENSE.oh-my-pi`。主迴圈沿用 mini-swe-agent 的直線流程。[Oh My Pi 基準版本](https://github.com/can1357/oh-my-pi/tree/24439066040e006eb08da7b204f0ac41c79d2de9) · [mini-swe-agent 迴圈](https://github.com/SWE-agent/mini-swe-agent/blob/main/src/minisweagent/agents/default.py)

## 執行

使用 Python 3.11+。先在環境中設定 `OPENAI_API_KEY`，再指定支援串流工具呼叫的 Chat Completions 模型。相容端點可透過 `OPENAI_BASE_URL` 設定。

```powershell
uv run mini_pi.py --model YOUR_MODEL --context-window YOUR_WINDOW_TOKENS --memory local --cwd 'E:\your-repo'
```

加上一個引號包住的任務字串即可單次執行。省略任務會進入互動模式，可用 `/compact`、`/shake`、`/memory`、`/continue`、`/exit`。用啟動時顯示的 UUID 搭配 `--session UUID` 載入 session，再加 `--continue` 可直接續跑未完成的那一輪。Session 與記憶預設存在 `~/.mini-pi/sessions.sqlite3`，可用 `--state-dir` 指定其他目錄。模型步數預設不設上限，可用 `--max-steps N` 限制每輪步數。

可用 `--max-time 10m` 設定本次啟動的工作階段期限，支援秒數及 `s`、`m`、`h` 單位。單次執行與 `--continue` 逾時時會取消當前工作並回傳非零狀態。互動終端、RPC 與 ACP 會中斷目前的回合，期限後的提示也不會再送到模型。終端和宿主可繼續操作已儲存的歷史。重開程序並指定同一個工作階段可接續執行。[上游 CLI 時間上限](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/cli-reference.md)

`shell` 可用 `async: true` 立即回傳工作 ID，完成後以 `async-result` 喚醒原工作階段。指令預設逾時 300 秒，可用 `timeout: 0` 取消工作自身的期限。未指定 `async` 的本機指令可依 `bash.autoBackground.thresholdMs` 從前景轉為背景，預設為 60 秒。用 `read proc://` 列出工作、`read proc://<id>` 查看狀態與輸出、`write proc://<id>/kill` 取消。`wait` 可直接收取結果，並取消該結果後續的自動交付。較長的輸出保存在工作階段的 artifact。[上游 shell 工具](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/tools/bash.md)

`shell` 的 `name` 可啟動同一工作目錄共用的具名服務。`ready` 可用 `log` 正規表示式、`port` 或兩者等待就緒。就緒逾時後服務會繼續執行。用 `read proc://<name>` 讀狀態及日誌、`write proc://<name>` 送入文字、`/kill` 停止、`/mode` 設定 `session`、`persist` 或 `detached`。`wait` 可等自己啟動的服務。服務自行退出或監督器異常退出時，會以 `launch-completion` 記入擁有者的歷史。重開同一工作階段會接收待交付的通知。手動停止或重啟只更新服務狀態。可用 `launch.enabled: false` 關閉。Windows 的預設服務偽終端由 PyWinpty 提供。 [上游服務執行](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/bash-tool-runtime.md#managed-service-execution)

具名服務優先採用 `shellPath`。Windows 預設探索 Git Bash，再回退至 cmd。Unix 優先採用可執行的使用者 Bash 或 Zsh。普通服務日誌保留當前與前一份，每份上限 25 MiB。監督器異常退出後，其他工作階段可辨識並清理一般服務的殘留程序樹，或接回仍在執行的分離服務。復原會比對 PID 和程序啟動時間。分離服務直接寫日誌，標準輸入不可用。 [上游 shell 選擇](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/utils/src/procmgr.ts) · [上游服務監督器](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/launch/broker.ts)

可用 `uv run mini_pi.py ps` 在 agent 外管理服務。終端開啟即時列表，可用方向鍵選取，Enter 查看詳情，`l` 看日誌，`s` 停止，`x` 強制停止，`r` 重啟，`a` 切換所有範圍，`q` 退出。靜態列表用 `ps --plain` 或 `ps --json`。`--all` 顯示同一狀態目錄的所有範圍，`--dir PATH` 指定其他工作目錄，`--global NAME` 指定已登記的全域範圍。 [上游管理指令](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/commands/ps.ts)

可在命令列執行 `ps info NAME`、`ps logs NAME --follow --grep REGEX`、`ps stop NAME --timeout 5`、`ps kill NAME` 和 `ps restart NAME`。重啟沿用原服務的命令、工作目錄、存續模式和工作階段歸屬。 [上游管理實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/cli/ps-cli.ts)

單次 CLI 加上 `--wait-async` 或設定 `async.printWake: true` 後，會持續處理背景工作及其喚醒回合，等工作都交付後才輸出最後回答。未啟用時，退出會取消仍在執行的背景指令。`--max-time` 也涵蓋等候背景結果的時間。[上游 headless 行為](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/cli-reference.md#headless--print-mode)

`--warn-cost 0.50` 在累計支出達到 0.50 美元時警告一次，`--max-cost 1` 在累計達到 1 美元時拒絕下一次模型請求。也可用 `session.warnCost` 和 `session.maxCost` 設定預設值。會以收到的用量和已知費率計入主回合、重試、壓縮、判斷與快取續期等請求，重開工作階段會讀回已記錄的支出。在途請求可以跨過門檻。單次 CLI 因上限停止會回傳狀態碼 `3`，JSONL RPC 的 `prompt_result` 會回報 `cost_cap`。[上游 CLI 成本控制](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/cli-reference.md)

`task` 預設接收 `context` 和 `tasks[]`。每個任務需要 `task` 與描述問題開放程度的 `solutionSpace`。子代理各有工作階段，可沿用 `modelRoles.task`，並共用祖先的成本上限。`task.maxConcurrency`、`task.maxRecursionDepth`、`task.maxRuntimeMs` 和 `task.softRequestBudget` 分別控制並行數、巢狀深度、執行時間與請求數。開啟 `async.enabled` 時，背景子代理的結果會經由既有的 `async-result` 交付。[上游 task 工具](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/tools/task.md)

`task.speculativeLaunch` 預設開啟。在允許直接派送的 CLI 和 RPC 回合中，共用的 `context` 和單一任務物件一封口，就可以開始該子代理。正式派送會重新驗證整批參數，再接手符合的執行並登記代理連結和背景工作。串流取消、呼叫被攔截或參數不符時，會取消並清理尚未接手的執行。`taskFile` 由正式派送讀取。可設定 `task.speculativeLaunch: false` 停用預啟動。 [上游串流預啟動](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/task/speculative-launch.ts)

可用 `task.batch: false` 改用單一任務格式，或用 `task.file: true` 讀取工作區內的任務檔。`.omp/agents/` 及使用者 profile 的 `agents/` 可定義專用代理和工具清單。設 `task.structured: true` 後可用 `outputSchema` 驗證子代理透過 `yield` 提交的結果。結果可用 `read agent://<id>` 或 `read agent://<id>/<field>` 讀取，歷史可用 `read history://<id>` 查閱。[上游代理定義](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/task-agent-discovery.md)

結構化子代理可用 `yield` 的 `type: ["欄位"]` 分段提交。陣列欄位逐項累積，純量欄位保留最近一次提交。`type: "result"` 可收尾並驗證整份結果，或用 `data` 提交一份完整結果。背景工作若在終止 `yield` 之後回傳，子代理需重新提交涵蓋新結果的報告。到達軟預算的 1.5 倍會要求立即收尾，最多允許五次額外請求。 [上游分段組裝](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/tui/src/tools/task-yield-assembly.ts) · [上游子代理執行器](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/task/executor.ts)

設 `task.isolation.enabled: true` 後，可用 `isolated: true` 將指定任務派到私人 Git checkout。未指定的任務使用父工作區。隔離任務會看到派工時的 staged、unstaged 和未追蹤檔案，結果只擷取相對於該快照的變更。`task.isolation.merge` 可選 `patch` 或 `branch`，`task.isolation.apply: false` 可只保留補丁與分支。衝突時會在結果附上補丁路徑，無法擷取時會保留工作區供恢復。Python 版使用本機 Git checkout，原生檔案系統快照後端仍待移植。[上游隔離行為](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/tools/task.md)

用 `write agent://<id>` 在同一工作階段樹內傳訊，子代理可用 `agent://Main` 回覆主代理。主代理可用 `agent://all` 廣播給執行中或仍在記憶體的閒置子代理。`?progress` 只回報進度，發問的代理會繼續等仍在線的收件者正式回覆。`wait` 發現對方必須等自己完成才有辦法回答時，會立即回報該依賴。訊息先寫入 SQLite，再於下一次模型請求前交付，或由 `wait` 直接取回。單次 CLI 可加 `--wait-async` 讓喚醒回合跑完。 [上游等待規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/tools/wait.ts)

子代理的執行環境預設保留七分鐘。`task.agentIdleTtlMs` 可調整閒置期限，設為 `0` 會保留至主工作階段關閉。暫停後收到新訊息會載入原本的工作階段。非隔離代理在主程序重開後也可喚醒。已交付的訊息在 JSONL 歷史以 `irc:incoming` 或 `wait` 工具結果保留。[上游代理生命週期](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/registry/agent-lifecycle.ts)

隔離代理在同一主工作階段內會沿用原本的私人 Git 工作區，包含閒置到期後的喚醒。釋放時會擷取初次交付後的改動並建立復原分支。主工作階段關閉並釋放工作區後，隔離代理的對話仍可查閱。 [上游隔離工作區生命週期](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/task/isolation-runner.ts)

也可用 `python -m pip install "openai>=2,<3" "httpx>=0.28,<1" "websockets>=15,<16" "filelock>=3,<4" "prompt-toolkit>=3,<4" regex pillow fonttools "mcp>=2.3,<3" "pyyaml>=6,<7" cryptography xxhash "jsonschema>=4,<5" "psutil>=7,<8" ast-grep-py wcmatch "pathspec>=0.12,<2" "resvg-py>=0.5,<1"` 安裝依賴，再以 `python mini_pi.py` 執行。Windows 上使用偽終端的具名服務另需 `pywinpty>=3,<4`。

工具包含 `read`、`write`、`edit`、`shell`。檔案工具使用 UTF-8，`edit` 要求原字串恰好出現一次。Shell 在指定工作目錄以目前帳號權限執行，Windows 使用 `cmd.exe`，其他平台使用 `/bin/sh`。每次指令最多等待 60 秒，工具結果會回傳至所設定的模型端點。

同一批的一般工具可並行，每個工作階段最多同時執行 32 個。寫檔、編輯、明確指定 `pty: true` 的 shell 與 `yield` 會獨占該批次的執行區間。結果完成即寫入 SQLite，傳給模型時仍按呼叫順序排列。轉向會在批次邊界送入，`wait` 可提早醒來。 [上游工具排程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/agent-loop.ts#L3908)

將 `tools.speculativeExecution.enabled` 設為 `true` 後，回覆中已封口的本機 UTF-8 文字讀取可提前執行。預設同時預讀兩個檔案，可用 `tools.speculativeExecution.maxInFlight` 調整。預讀限工作目錄內、允許直接讀取、最多 4 MiB 的檔案。若前面有不符合條件的工具呼叫，後面的讀取會等正式派送。採用結果前會重新確認呼叫、路徑和內容摘要。檔案有變就照常讀取。串流重試、取消或 TTSR 中斷時會捨棄預讀結果，正式派送時才寫入工作階段。 [上游推測執行](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/speculative-execution.ts)

## `eval`

`eval` 每次執行一個 `{"language":"py","code":"..."}` cell。同一工作階段會保留變數和匯入，支援 top-level `await`、尾端運算式、`%load`、`%pip` 和 `await tool.<name>({...})`。可用 `reset: true` 重建 kernel，或將 `python.kernelMode` 設成 `per-call` 以每次啟動新 kernel。預設逾時 30 秒，可用 `timeout: 0` 關閉該 cell 的期限。輸出過長會存到 artifact。 [上游 Python REPL](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/python-repl.md)

`eval.autoBackground.enabled: true` 會在 cell 超過 `eval.autoBackground.thresholdMs` 後改由背景工作交付，預設門檻為 60 秒。上下文壓縮會附上活躍 kernel 與已載入腳本的摘要。重新啟動程序後，需重新執行變數或匯入的設定。可用 `eval.py: false` 或 `PI_PY=0` 隱藏 Python `eval`，用 `python.interpreter` 指定執行檔。 [上游 eval 工具](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/tools/eval.md)

JavaScript cell 使用 `{"language":"js","code":"..."}`，由獨立的 Bun worker 執行。支援 top-level `await`、相對路徑的 JS 和 TypeScript 匯入，以及 `%load`。JavaScript 與 Python 各自保留狀態，`reset` 只會重建該語言的 runtime。用 `%bun add` 安裝套件，預設存入同一專案共用的受管理環境。可用 `%environment project` 讓套件安裝寫入專案，用 `%environment managed` 切回。套件安裝停用 lifecycle scripts。 [上游 JavaScript runtime](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/tools/eval.md#javascript-js)

JavaScript 需要 Bun 1.4.4+。將 Bun 放在 PATH，或用 `PI_BUN` 指向執行檔，然後在 Mini Pi 根目錄安裝已鎖版的 worker 依賴。可用 `eval.js: false` 或 `PI_JS=0` 停用。 [Bun 安裝文件](https://bun.com/docs/installation)

```powershell
bun install --cwd eval_kernel/js --frozen-lockfile --ignore-scripts
```

Python 和 JavaScript 的 `completion()` 會啟動無對話歷史的模型請求，立即回傳 handle。可用 `model` 選 `smol`、`default` 或 `slow`，以及用 `system` 和 `schema` 設定提示和結構化回覆。handle 可跨 cell 等待，支援狀態、備援嘗試紀錄和取消。`wait()` 按傳入順序回傳結果，等待時會暫停 cell 的逾時計時。completion 的用量與主回合共用工作階段的成本上限。 [上游 completion 實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/eval/completion-bridge.ts)

`await judge(state, questions)` 依 `modelRoles.judge` 和 `retry.fallbackChains.judge` 選模型，回傳各題的 `bool`、`choice` 或 `score`。Python 可傳 `images=[...]`，JavaScript 可傳 `{images: [...]}` 作為第三個參數。圖片支援本機檔案、SVG 和內嵌資料。原生 API 保留機率分布並逐題快取，文字模型用單次請求判斷同一狀態的多題。等待會暫停 cell 的逾時計時，用量計入工作階段成本。 [上游 judge 實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/eval/judgment-bridge.ts)

Python `judge_batch()` 和 JavaScript `judgeBatch()` 以相同題目判斷一批狀態。批次可指定逐項圖片、並行度、重試次數與最低成功數。`drain()` 依完成順序讀取新增結果。重設 kernel 後，可用 `attach(id)` 接續同一個游標。`status()` 顯示進度和費用，`failed()` 保留各項錯誤，`cancel()` 中止批次，`close()` 釋放批次。啟用背景工作時，工作完成會自動交付摘要，也可用 `proc://<id>/kill` 取消。 [上游批次實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/eval/judgment-batch-bridge.ts)

Python 和 JavaScript 的 `agent()` 會建立背景子代理。`wait()` 可以混合子代理與 completion handles，按輸入順序回傳結果。已由 `wait()` 取得的子代理結果會從背景交付佇列移除。可用 `handle.output()` 或 `agent://<id>` 讀取輸出，用 `handle.send()` 追問同一子代理。若允許每次呼叫選模型，`model` 必須是操作者已授權的具名模型。可指定 `schema`，也可用 `isolated: true` 將改動留在私人 Git 工作區，讓 `apply` 和 `merge` 控制回傳方式。 [上游 agent 實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/eval/agent-bridge.ts)

用 Python `@tool` 或 JavaScript `tool(fn, options)` 註冊函式後，可把名稱列在 `agent()` 的 `tools` 選項，或普通 `task` 的 `tools` 欄位。子代理會回呼父工作階段的 kernel，即使父 cell 正在 `wait()`。兩個 kernel 註冊同名工具時，須先取消其中一個註冊。重設 kernel 會清除註冊，`eval.tools.enabled: false` 可停用工具授權。 [上游自訂工具實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/task/eval-tools.ts)

Python `workpool()` 和 JavaScript `await workpool()` 可用 `push()` 將多項工作交給保留對話的子代理。忙碌 worker 輪流收取排隊項目，下一回合可一起處理多項。若有閒置 worker，優先使用上下文占比最低的。worker 透過 `yield(key=…, data=…)` 或 `yield(key=…, error=…)` 逐項提交，最後一個 key 結束該回合。`status()` 顯示進度，`peek()` 可重複讀取批次結果。pool 清空時會向主回合交付一次總結果，可用 `wait` 工具等待。`close()` 取消排隊項目，`proc://<pool>/kill` 也會中止正在執行的批次。`eval.workpool.freshAgents: true` 讓每項工作使用新的 worker，`task.maxConcurrency: 0` 取消 pool 的 worker 數量上限。 [上游 workpool 實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/task/workpool.ts)

開啟 `tools.speculativeExecution.enabled` 後，CLI 和 RPC 可用閒置 runtime 的快照規劃串流中的 Python 和 JavaScript cell，提前讀取符合條件的 `await tool.read(...)`。JavaScript 規劃在獨立 Bun 程序解析 AST。cell 開始前會原子比對 namespace 的版本和摘要，採用預讀時也會比對呼叫位置、參數和檔案內容。依賴另一筆讀取結果的路徑須等該結果正式採用才會預讀。自動背景、重設及每次新建 kernel 的 cell 走一般執行。JavaScript 主 worker 啟動失敗時，Python 會嘗試在 Bun 子程序內改用 Worker 執行緒。 [上游 eval 推測執行](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/eval/speculation/cell-session.ts)

`read` 可在原 session 讀取 `artifact://<id>`，長內容會附上下一頁的 `?offset=` 引用。Shell 的 UTF-8 文字輸出超出顯示上限時會保存在 artifact。超過 16 MiB 時保存開頭與結尾，並標明省略的位元組數。

互動終端預設在一輪完成且閒置 240 秒後顯示一次簡短回顧。`--idle-recap-seconds 0` 可關閉。回顧存入 SQLite 的 `session_recaps` 表。可用 `--idle-compaction` 啟用閒置壓縮，預設門檻為 200,000 token，延遲 300 秒，分別由 `--idle-threshold-tokens` 和 `--idle-compact-seconds` 調整。輸入草稿或背景工作會延後排程。若閒置壓縮已經開始，下一個 prompt 會在壓縮完成後處理。供應商確認仍有效的一小時快取會讓壓縮提早至續期時間前五秒。其他情況使用設定的延遲。[上游閒置維護](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md#overflowincomplete-recovery-vs-thresholdidle-maintenance)

## 串流期間的輸入

在互動終端，主回合串流時按 `Enter` 可送出轉向訊息，`Ctrl+Q` 則把追問排在目前回合後。`Alt+↑` 或 `Shift+↑` 可取回最近一筆待送文字。`Esc` 會中止主回合，並把尚未送入對話的文字放回編輯區。可用 `steeringMode`、`followUpMode` 選擇 `one-at-a-time` 或 `all`。轉向會保留正在進行的檔案與 shell 操作，並喚醒 `wait`。 [上游終端輸入](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/controllers/input-controller.ts)

主回合期間按 `Ctrl+Q` 送出 `/compact [模式] [重點]`，會等較早的訊息和背景工作處理完再壓縮。按 `Enter` 會中止目前回合並立刻壓縮，成功後可依 `compaction.autoContinue` 接續未完成的工作。可選 `soft`、`remote` 或 `snapcompact`。 [上游手動壓縮](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/slash-commands/builtin-lifecycle.ts#L228)

## 串流規則 TTSR

在 `.omp/rules/` 放 Markdown 規則。YAML frontmatter 可用 `condition` 設定正規表示式、`astCondition` 設定原始碼模式，或用 `question` 讓裁判判定。`scope` 預設比對回覆與工具參數，也可選 `thinking` 或 `tool:write(*.py)` 等範圍。規則亦可用 `agents` 限定代理。一般規則可加 `description`，讓模型透過 `rule://<name>` 讀取正文。 [上游規則發現與範圍](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/rulebook-matching-pipeline.md)

Chat Completions、Responses 與 Anthropic Messages 的文字、思考與工具參數可在串流中觸發規則。`ttsr.contextMode` 預設 `discard`，也可設定 `keep` 保留中斷時的部分輸出。`interruptMode: never` 會讓工具照常執行，並在工具結果之後追加獨立的開發者提醒。AST 在工具執行前比對完整原始碼片段。注入紀錄隨工作階段分支保存，支援 `ttsr.repeatMode` 的 `once` 與 `after-gap`。 [上游注入生命週期](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/ttsr-injection-lifecycle.md)

`ttsr.judge` 預設 `auto`，在裁判路由採用 TypeSafe 時檢查完成的輸出。`on` 也可使用文字裁判，`off` 關閉。相同輸出的問題會合併請求，命中的規則以補充提醒送回。內附上游同一版本的 26 條規則，可用 `ttsr.builtinRules: false` 關閉，或透過 `ttsr.disabledRules` 排除個別規則。

`ttsr list` 列出已註冊的規則。`ttsr test` 可重播文字、檔案或標準輸入，搭配 `--agent` 和 `--rule` 檢查代理範圍或隔離測試單一規則。`ttsr scan` 以檔案工具範圍掃描專案，預設遵循 `.gitignore`，並略過二進位、超過 5 MiB 的檔案及裁判規則。三個指令皆支援 `--json`，並與 agent 共用 `.omp/settings.json` 和 `.omp/config.yml` 的 TTSR 設定。 [上游 TTSR 指令](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/cli/ttsr-cli.ts)

```powershell
uv run mini_pi.py ttsr list
uv run mini_pi.py ttsr test --path src/example.py "print(value)"
uv run mini_pi.py ttsr scan src --json
```

在互動終端輸入 `/omfg <抱怨>`，目前的模型會在輸入區上方串流顯示候選規則，並以所選分支的歷史輸出驗證。生成時仍可編輯或提交主對話，按 `Esc` 會取消側請求並保留草稿。長規則可用 `PgUp`、`PgDn` 檢視。你可以要求修改、存到專案或存到目前 profile。未確認命中的規則會另外詢問是否儲存。生成與裁判請求不會增加對話訊息。 [上游 `/omfg` 流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/controllers/omfg-controller.ts)

## 重複執行 `/loop`

在互動終端輸入 `/loop 3 review the failing tests`，會先送出提示，再自動重送最多三次。可用 `10m` 或 `1h30m` 改設時間上限。省略提示時，迴圈會等下一次手動送出。按 `Esc` 暫停，手動送出新提示可接續並替換重送內容。再輸入 `/loop` 關閉。 [上游排程與暫停](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/interactive-mode.ts#L3303)

`/loop 10 --until 'pytest -q' fix the failing tests` 在每次重送前執行條件。`--until` 在退出碼 `0` 時停止，`--while` 在退出碼 `1` 時停止。其他退出碼和逾時會報錯並停止。首輪直接執行。`loop.conditionTimeoutMs` 預設 `30000`，`0` 代表無限期等待。使用 Bash 或其他支援的 POSIX shell 時，同一會話會保留條件 shell 的變數和函式，每次檢查從會話工作目錄開始。Bash 的頂層 `exit` 或 `set -e` 結束檢查後，下次可恢復變數、函式與選項。函式內退出或自訂 `EXIT` trap 可能使 Bash 直接重建。取消、逾時或切換會話會清除條件 shell 的狀態。PowerShell 和 `cmd.exe` 等其他 shell 每次檢查會各自啟動。 [上游條件判定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/loop-condition.ts)

`loop.mode` 可設為 `prompt`、`compact` 或 `reset`，依序表示直接重送、手動壓縮後重送、開新會話後重送。`/loop` 開啟期間會暫停 `/goal` 的自動接續。 [上游重送模式](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/settings.ts#L992)

## 長程目標 `/goal`

在互動模式輸入 `/goal <目標>` 開始，或用 `--goal "<目標>"` 開一個新的互動會話。`/guided-goal [初步想法]` 先讓模型釐清需求與驗收證據。可用 `/goal show`、`pause`、`resume`、`drop`，或用 `/goal budget N` 設定 token 預算。`/goal budget off` 清除預算，`/goal set <目標>` 可替換活動中的目標。目標狀態寫入會話分支，重新開啟程序後可用 `/goal resume` 接續。 [上游目標狀態與預算](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/goals/runtime.ts)

預設只有互動模式會自動續跑。RPC 可設定 `goal.continuationModes: [interactive, rpc]`，用 `{"type":"goal","op":"create","objective":"…","token_budget":50000}` 建立目標，並用 `get`、`pause`、`resume`、`drop` 管理。預算計入新輸入、輸出及快取寫入，快取命中不計入。續跑若沒有工具活動或重複同一組工具與結果，就會停在目前進度。預算耗盡會請模型收尾，模型驗證完全部交付項目後可透過 `goal` 工具回報完成。 [上游 RPC 續跑控制](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/rpc/rpc-goal.ts)

串流期間可直接執行上述 RPC 目標操作，終端也可用 `/goal set <目標>` 或 `/goal budget N` 即時調整。新目標會在模型回合邊界送入。暫停或捨棄會立即更新狀態。若舊回應嘗試完成已替換的目標，工具會要求先讀取目前狀態。 [上游串流中的目標操作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/interactive-mode.ts#L7636)

### 階段式待辦

`todo` 工具依階段保存工作，以任務原文定位項目。完成、移除或阻塞目前項目後，工具會自動選取下一個可執行項目。`/todo` 可查看，`/todo edit` 可在互動終端編輯，`/todo export` 與 `/todo import` 可用 Markdown 交換清單。RPC 可在 `get_state.todoPhases` 讀取，並用 `set_todos` 更新。待辦會隨分支回溯與會話重開恢復。只剩阻塞項目時，`/goal` 會等使用者或外部工作推進。 [上游待辦工具](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/tools/todo.ts)

模型收尾時若還有可執行的待辦，系統會送出有限次提醒。等待使用者或背景工作的回合會延後提醒。`todo.reminders: false` 可關閉提醒，`todo.remindersMax` 可調整收尾提醒的上限。`todo.eager` 可設為 `preferred` 建議首次先建清單，或設為 `always` 在支援指定工具的模型上要求先呼叫 `todo`。 [上游待辦提醒](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/todo-tracker.ts)

互動終端在輸入區上方顯示待辦 HUD。收合時保留目前階段、最近一項已結束工作和後續摘要。`/todo expand` 展開全部，`/todo collapse` 恢復摘要。低於 18 列的終端改用單行進度。相符的子代理工作會標為進行中，成功結束時會更新對應待辦。全部結束後，HUD 預設 60 秒後隱藏，清單仍可用 `/todo` 查看。可用 `tasks.todoClearDelay` 調整延遲，`-1` 表示持續顯示。隱藏和重新顯示都記在該筆待辦的分支日誌。 [上游 HUD 與自動隱藏](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/interactive-mode.ts#L5163)

## 側對話 `/btw`

輸入 `/btw <問題>` 可在目前上下文中開啟側對話。終端會串流顯示答案，也可同時編輯或提交主對話。單獨輸入 `/btw` 開啟歷史，用方向鍵和 `Enter` 瀏覽，`f` 追問，`c` 複製答案。`Esc` 可取消進行中的請求，並保留已收到的答案。歷史按工作階段保存在 `artifacts/<session>/btw-history/`。 [上游側回合生命週期](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/btw-turn.ts)

單輪側對話完成後，若主工作階段仍停在開始時的節點，按 `b` 可建立分支，把原始問題和答案接入該分支。將 `btw.inject` 設為 `true` 後，帶有完整 `<inject>` 區塊的新答案可用 `Alt+I` 送出指示。RPC 用 `btw`、`btw_cancel` 和 `get_btw_history` 操作，同一主題的追問可傳 `recordId`。串流與狀態分別由 `btw_delta` 和 `btw_record` 回報。 [上游 RPC](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/modes/rpc/rpc-btw.ts)

## 模型路由

用 `--route-config` 指向 JSON 檔，再用 `--model` 選擇主路由的別名。每條路由可以指定模型、上下文視窗、推理強度、端點和憑證環境變數。端點未設定時沿用 `OPENAI_BASE_URL`。指定 `base_url` 時可用 `api_key_env` 或 `broker_provider` 供應憑證。`api_key_env` 可以填單一變數名稱或依順序填多個名稱。

Python 內附同一 commit 的 `catalog/models.json` 和 `catalog/rules.json`。用 `uv run mini_pi.py --list-models 'anthropic/*'` 可列出模型。用 `--model anthropic/claude-fable-5-1` 可直接選目錄模型，沿用供應商的環境變數、上下文視窗和輸出上限。支援直接路由的目錄 API 包含 Chat Completions、Responses 和 Anthropic Messages。[上游模型設定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md)

OpenRouter 和 Vercel AI Gateway 可用 `provider/model@upstream:level` 指定上游路由。完整的目錄 ID 優先，因此 ID 內原有的 `@` 仍可使用。`--model`、模型角色和備援鏈都可使用這個格式，同一模型的不同 upstream 會分開記錄。子代理的 `model` 需匹配操作者授權的精確路由，包括 pin。工作區可進一步限制允許清單，恢復子代理時會重查授權。 [上游模型解析](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#runtime-model-resolution) · [上游子代理授權](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/task-agent-discovery.md#per-call-model)

Python 會讀取目前 profile 的 `agent/models.yml`、`models.yaml`，以及舊版 `models.json`。預設 profile 支援 `PI_CODING_AGENT_DIR`。可用 `--models-config PATH` 指定檔案。自訂模型可直接用於 `--model`、角色和備援。模型設定依 provider、`models`、`modelOverrides` 合併。啟用 `extendedContext` 後會採用模型指定的較大 `maxContextWindow`。可用 `auth: none` 連接免金鑰端點，或以環境變數、字面值及 `!指令` 提供金鑰。指令在工作目錄執行，結果快取於程序中。收到 401 時會刷新指令產生的憑證，若值有變更則重送一次。[上游模型設定合併順序](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#merge-and-override-order)

`compactionModel` 指定文字摘要使用的首選模型，可用同 provider 的模型 ID 或完整的 `provider/model`。Responses 模型也可用 `remoteCompaction` 啟用原生壓縮，指定 V1、V2 的同源端點和請求用模型 ID。

可用 `discovery.type: ollama`、`llama.cpp`、`lm-studio`、`openai-models-list` 或 `proxy` 取得端點上的模型。未設定 Ollama 時，預設依 `OLLAMA_BASE_URL`、`OLLAMA_HOST`、本機 11434 埠的順序找端點，透過 Responses 發送推理請求。用 `uv run mini_pi.py --list-models 'ollama/*' --refresh-models` 重新探索。快取位於 state 目錄的 `models-cache.sqlite3`，以端點與請求憑證區隔。Ollama、LM Studio、OpenAI 相容及混合代理可直接使用 24 小時內的快取。重新探索失敗時則沿用上次的模型資料。供應商憑證和請求標頭值只在記憶體解析。[上游模型探索](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#runtime-discovery-integration)

llama.cpp 預設使用 `LLAMA_CPP_BASE_URL`，未設定時先試本機 8080，連不上再試 9931。啟動時探測 `/models` 與 `/props`。LM Studio 預設使用 `LM_STUDIO_BASE_URL` 或本機 1234，從 `/v1/models` 取得清單，並讀取兩代原生端點的中繼資料。模型選取時和首次完成推理後會更新實際容量與輸入能力，並重算壓縮門檻，明確設定的模型或路由限制優先。探索到的 llama.cpp Qwen 模型使用 Chat Completions 與其聊天模板控制。同一部署的推理歷史會在後續請求重放，並計入上下文預算。[上游 llama.cpp 與 LM Studio 行為](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#implicit-llamacpp-discovery)

需要 STB 解碼器的模型會在送出請求時把 WebP 轉成 PNG 或 JPEG。歷史訊息仍保留原始圖片。LM Studio 的 Decisions 模型會以 `openai-decisions` 顯示在模型清單，可設定為 `modelRoles.judge`。

JSON 路由可加 `"catalog": "openai/gpt-5.4"` 繼承目錄資料，並用路由欄位覆寫。自訂端點需明確指定憑證。Oh My Pi profile 設定 `contextPromotion.enabled: true` 時，Python 也會採用目錄指定、可用且視窗更大的晉升目標。`promote_to` 優先。[上游上下文晉升](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#context-promotion)

省略 `--model` 時會嘗試 profile 的 `modelRoles.default`。也可以用 `--model '@slow:high'`。角色可引用其他角色或指定逗號分隔的候選，啟動時選第一個有憑證的候選。自動記憶可用 `modelRoles.memory`，未指定時會查看已設定的 `tiny` 或 `smol`。明確的 `--maintenance-model` 優先。

互動工作階段收到第一個具體任務後會在背景命名。`title.generator: fork` 使用主回合的模型與請求前綴，`title.generator: tiny` 使用 `tiny`、`commit`、`smol` 的線上模型與備援。可在 `.omp/TITLE_SYSTEM.md` 自訂命名提示，或設定 `PI_NO_TITLE=1` 停用自動命名。`/rename <標題>` 可手動命名，RPC 可用 `set_session_name`。SQLite 和 JSONL 都會保存標題。 [上游命名流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/utils/title-generator.ts)

設定 `memory.backend: local` 會啟用 SQLite 與 Markdown 檔案記憶儲存，摘要請求透過已設定的模型端點執行。工作階段可用 `--memory off` 停用。`memories.minRolloutIdleHours`、`memories.stage1Concurrency` 和 `memories.summaryInjectionTokenLimit` 分別控制擷取等待、並行度和注入預算。擷取前及寫入摘要前會遮蔽符合憑證格式的字串。明確選用 `local/<model>` 而沒有相容端點時，相關的模型請求會跳過。 [上游模型角色與本機模型說明](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/local-models.md)

用 `read memory://root` 讀取專案摘要。`memory://root/MEMORY.md`、`learned.md`、`rollout_summaries/` 和 `skills/` 可供進一步查閱。`autolearn.enabled: true` 會開啟 `manage_skill`，本機記憶開啟時也提供 `learn`。新經驗立即寫入 `learned.md`，下一個工作階段才會將它注入提示詞。`autolearn.autoContinue: true` 可在達到 `autolearn.minToolCalls` 後執行獨立的擷取回合。`/memory` 支援 `view`、`stats`、`diagnose`、`queue`、`sync`、`enqueue`、`rebuild` 和 `clear`。 [上游記憶工具](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/memory.md) · [上游 Auto-Learn](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/autolearn/controller.ts)

`memory.backend: hindsight` 使用 `hindsight.apiUrl` 指定的記憶服務，預設為 `http://localhost:8888`。可用 `HINDSIGHT_API_URL` 與 `HINDSIGHT_API_TOKEN` 指定端點與憑證。它支援首回合回憶、按回合保存對話，以及 `retain`、`recall`、`reflect` 和選用的 `learn`。工具寫入會先排入本機佇列，在服務接受後移除。預設以專案標籤隔離，開啟 `hindsight.globalScope` 才能主動寫入跨專案記憶。`/memory mm` 可管理服務端的 mental models。 [上游 Hindsight](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/memory.md#hindsight-remote-backend)

Hindsight 與 Mnemopi 在代理回合結束後檢查使用者回合數，達到各自的保存間隔才擷取對話。恢復工作階段後，首次回憶只讀取查詢需要的最近回合。Hindsight 保存完整對話時會驗證並重用前次已保存的前綴。

`memory.backend: mnemopi` 使用 BEAM 格式的 SQLite 記憶銀行。支援 `retain`、`recall`、`reflect`、`memory_edit` 和 `read memory://<id>`。預設為 `per-project`，`per-project-tagged` 會同時查詢本專案與共用銀行。工作階段記錄會保存第一次注入的回憶，恢復時另外交付撤銷或更新提醒。`/memory sync` 將到齡的工作記憶整併成情節。預設可用全文搜尋。需要向量召回時，可用 `mnemopi.embeddingApiUrl` 和 `mnemopi.embeddingModel` 指定受信任的 embeddings 服務。 [上游 Mnemopi](https://github.com/can1357/oh-my-pi/tree/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi)

Mnemopi 的 `llmMode: smol` 透過 `memory` 模型角色在背景擷取事實。`remote` 使用設定的 `llmBaseUrl` 和 `llmModel`，`none` 使用文字規則。自動保存對話時只把使用者訊息交給擷取器。結構化結果分別寫入事實、指令、偏好、時間線和三元組。情節整併會寫入摘要節點與關係，`mnemopi.proactiveLinking: true` 則在保存工作記憶時建立連結。 [上游擷取流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi/src/core/extraction.ts) · [上游情節圖譜](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi/src/core/episodic-graph.ts)

`mnemopi.polyphonicRecall: true` 會融合混合搜尋、向量、圖譜、事實與時間五路排名，並可用 `MNEMOPI_POLYPHONIC_RECALL` 或各路的 `MNEMOPI_VOICE_*` 環境變數覆寫。`mnemopi.enhancedRecall: true` 啟用四層查詢快取。快取依銀行與查詢設定隔離，並在偵測到 SQLite 有其他連線寫入時失效。 [上游多路召回](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi/src/core/polyphonic-recall.ts) · [上游查詢快取](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi/src/core/query-cache.ts)

開啟已設定 embeddings 服務的銀行時，背景工作會補齊缺少的向量。變更 `mnemopi.embeddingModel` 時，先測試新模型，再分批產生向量，整批成功後以單筆 SQLite 交易更新。途中失敗會保留舊向量，下次開啟時再試。`/memory sync` 可等待背景工作。`MNEMOPI_EMBEDDING_BATCH_SIZE` 與 `MNEMOPI_EMBEDDING_MAX_INPUT_CHARS` 可調整請求批量和輸入長度。 [上游向量重建](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi/src/core/beam/store.ts) · [上游 embeddings 請求](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi/src/core/embeddings.ts)

`/memory export <file.json>` 以 BEAM 1.0 格式備份目前寫入的銀行。`/memory import <file.json>` 預設跳過重複的工作與情節記憶。可用 `/memory import --force <file.json>` 覆寫。相對路徑以工作目錄為準。若銀行已包含舊版 `vec_episodes` 虛擬表，可用 `uv run --with sqlite-vec --script mini_pi.py …` 載入擴充套件以匯入舊版情節向量，支援 float32、int8 和 bit。一般匯出依 BEAM 1.0 格式輸出空的 `episodic_embeddings`。 [上游備份與還原](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/mnemopi/src/core/beam/store.ts#L923) · [sqlite-vec Python 文件](https://alexgarcia.xyz/sqlite-vec/python.html)

`memory.backend: sharpshooter` 使用 `sharpshooter.model` 或 `smol` 角色擷取專案決策。擷取會檢查引文確實出現在當前提示，排入工作階段佇列。整併以原始決策、既有記憶與專案文件更新 `architecture.md`、`product.md` 和 `style.md`。`/memory sync` 可立即整併，`/memory queue` 可查看待處理的決策。 [上游 Sharpshooter](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/memory.md#sharpshooter-backend)

`retry.fallbackChains` 支援 `provider/model:effort`、`provider/*`、`provider/prefix/*` 和角色鍵。備援可指定推理強度，省略時沿用失敗回合的強度。`retry.fallbackOn`、`retry.modelFallback` 和 `retry.enabled` 控制是否重試。冷卻到期後會在下個使用者回合嘗試恢復，`retry.fallbackRevertPolicy: never` 則留在備援。拒答後選中的備援會在目前工作階段保持。[上游角色](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#role-aliases-and-settings) · [上游備援設定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/non-compaction-retry-policy.md#controls-settings-and-rpc)

`features.unexpectedStopDetection` 預設為 `mechanical`，會針對有簽章或加密推理但無可見輸出的回合續跑，最多三次。設為 `smart` 也會評估有可見文字的停止。判斷會使用 `modelRoles.judge` 與 `retry.fallbackChains.judge`，支援 OpenAI Decisions、TypeSafe、OpenRouter Decisions、Cloudflare Workers AI 的文字判斷，以及提示式聊天模型。進入原生判斷後，備援限於其他原生判斷模型。判斷用量會記入 session，成功的原生答案可在下次使用本地快取。設為 `none` 可關閉這項偵測。[上游判斷模型流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/environment-variables.md#typesafe-judgments)

`defaultThinkingLevel: auto`、`--reasoning-effort auto` 或 RPC 的 `set_thinking_level` 可啟用自動推理強度。新的使用者提示會透過 `judge` 角色評估，預設上限為 `xhigh`，可設定 `providers.autoThinkingMaxEffort: max`。已有唯一可用強度時直接套用。判斷在四秒內無法完成時，沿用最近的具體強度或模型暫定值。同一輪的工具續跑、備援和 `--continue` 沿用已判定的強度，並依當前模型的支援範圍調整。獨立出現在正文的 `ultrathink` 可直接選最高支援強度，`chillax` 可將支援逐訊息調整的模型降至最低強度。[上游強度控制](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/model-controls.ts#L655)

`broker_provider` 指定模型路由使用的 Auth Broker provider ID。憑證順序為明確設定的 `api_key_env`、OAuth、登入金鑰、預設環境變數、其他已存金鑰。`broker_accounts` 可用 `identityKey` 清單把路由限制在指定的 OAuth 帳號，且須省略明確的 `api_key_env`。程序也可透過 `OMP_AUTH_BROKER_ACCOUNT_POOL_FILE` 限縮可見帳號及金鑰。收到 401 時先嘗試刷新同一 OAuth 帳號並重送一次。收到 429 時會將限流狀態寫回 broker，再嘗試其他可用憑證。`broker_scope` 可指定限流作用範圍。Broker 憑證只會送到路由設定的 HTTPS 或 HTTP loopback 端點。[上游憑證選擇與帳號池](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/auth-broker-gateway.md#client-account-pools)

Anthropic 與 Codex 的 OAuth 帳號可依 broker 的配額報告排名。選擇時會避開已耗盡的適用週期，並比較短週期用量、長週期重置時間與 Codex 方案資格。Codex 的 chat 與 Spark 各有自己的計量範圍。可在 Oh My Pi 設定檔用 `auth.accountPolicies` 設定帳號的 `priority`、`reservePct` 和 `reclaimAbovePct`，以 `retry.usageReservePct` 設定啟用帳號政策時的預設保留比例。Broker 使用量依 provider 快取 15 秒，同一 provider 的並行查詢共用一次請求。[上游配額排名](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth/rank.ts) · [上游 Broker 設定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/auth-broker-gateway.md)

Anthropic 與 Codex 的回應標頭可立即更新對應帳號的配額週期。這些資料在程序記憶體保留 15 秒，即使 broker 暫時沒有完整報告也能用於排名。同一帳號的一般更新相隔至少 60 秒，耗盡訊號立即生效。Codex 成功回應的 100% 用量視為警示。[上游標頭快取](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth-broker/remote-store.ts)

排名發現配額耗盡時，會把對應帳號與範圍封鎖到供應商報告的重置時間。未提供重置時間時，預設封鎖 60 秒。封鎖超過五分鐘後，新鮮且涵蓋相關計量範圍的健康報告可讓帳號提早恢復使用。Anthropic 帳號級解封需要完整的共享週期資料。Codex 依 chat 和 Spark 各自的計量狀態判定。認證與帳號政策封鎖持續由各自的規則管理。[上游封鎖與恢復](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth/blocks.ts#L504)

使用 Auth Broker 的模型路由會將 agent 回合已收到的 token 計數每 10 秒批次送出。回報包含主機名稱、共用的 `install-id`、`OMP_APP_NAME`（預設 `omp`）、供應商與模型。計數分為一般輸入、輸出、快取讀取和快取寫入。可用 `cache_prices` 估算美元費用，或採用 OpenRouter 與 Cline Pass 回傳的實際費用。暫時失敗會保留批次供下次重送，正常退出最多等候兩秒送出餘量。[上游用量回報協定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/auth-broker-gateway.md#endpoints)

沒有設定 Auth Broker 時，指定 `broker_provider` 的模型路由會嘗試讀取目前 Oh My Pi profile 的 `agent.db`。也可用 `--agent-db PATH` 明確選擇本機庫。支援 schema v8 的現有 API key、OAuth 權杖、配額快取與作用域封鎖。封鎖直接寫回同一份 SQLite 資料庫，後續請求會讀到其他程序的變更。[上游 SQLite 憑證庫](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth/sqlite-credential-store.ts)

本機 Anthropic 與 Codex OAuth 會在接近到期或收到 401 時嘗試換發。Python 和 Oh My Pi 使用同一張 SQLite 租約表。只有仍持有租約、且資料列仍是換發前版本的程序能寫回新權杖。單次呼叫最多等十秒。已送出的換發會在背景完成，正常退出會等待寫回。供應商明確拒絕且沒有舊租約持有人可能仍在存檔時，會停用該版本的憑證。[上游續期流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth/refresh.ts)

`usage reset` 列出已登入的 Claude 與 Codex 帳號及可用額度。加上帳號 ID 會兌換一份。可在互動終端輸入相同的 `/usage reset` 命令，也可用 `--agent-db PATH` 直接操作本機資料庫。Python Auth Broker 在服務端處理兌換。連到原版 Auth Broker 時，客戶端會用 Broker 核發的 access token 向供應商查詢和兌換，再透過 Broker 使配額快取失效並更新封鎖。[原版 Broker 協定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth-broker/server.ts)

```powershell
uv run mini_pi.py usage reset
uv run mini_pi.py usage reset claude/12 --credit-id cedar-grant-id
uv run mini_pi.py usage reset codex/23
```

兌換前會取得與原版 Oh My Pi 共用的帳號鎖並重讀供應商額度。Claude 依供應商指定的 Cedar 或 Juniper 額度兌換，Codex 預設選最早到期的一份。已送出但結果不明的 Cedar 和 Codex 請求會保留原請求 ID，Juniper 結果不明時會等待額度狀態改變。兌換確認後會使配額快取過期，並根據受影響的計量範圍解封。Claude 只解除有證據已恢復的配額封鎖，Codex 解除觀察到的配額封鎖。兩者都保留認證與帳號政策封鎖。[上游帳號鎖](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/reset-fence.ts) · [上游 Claude 與 Codex 兌換](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth/resets.ts)

使用原版遠端 Broker 時，帳號鎖和未確認的請求記錄都在客戶端 profile。共用 profile 的同機程序會互斥，不同主機各自持有自己的鎖。客戶端每次刪除封鎖前會重讀 Broker 快照；原版協定的刪除端點只接受作用域，無法在服務端比對封鎖版本後再刪除。[上游快照與封鎖端點](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth-broker/server.ts)

自動兌換讀取 Oh My Pi profile 的 `claudeResets` 與 `codexResets` 設定。收到 429 後先嘗試換帳號。可用帳號都受限時，才依適用窗口、自然解封時間與額度保留量決定是否兌換，成功後重試當前回合。`autoRedeem: "yes"` 啟用、`"no"` 停用。未設定時，互動終端會在受阻回合詢問，選擇記在 Mini Pi 的狀態資料庫。無互動介面的執行只會自動使用五分鐘內到期的適用額度。等待門檻 `minBlockedMinutes` 預設 60、`keepCredits` 預設 0、`salvageHorizonHours` 預設 12。[上游 Claude 規劃](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/claude-auto-reset.ts) · [上游 Codex 規劃](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/codex-auto-reset.ts)

互動終端每五分鐘查詢配額。未設定自動兌換時，可用且快到期的額度會等輸入框空白後詢問。送出兌換前會重讀供應商額度，並核對原本顯示的額度與餘量。六小時內到期且涵蓋已使用至少 25% 配額窗口的額度會顯示一次提醒，即使供應商尚不允許兌換。本機與 Oh My Pi 共用 `agent.db` 的提醒記錄。透過遠端 Broker 時，提醒記錄放在客戶端 Mini Pi 狀態庫。[上游到期提醒](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/reset-expiry.ts)

```json
{
  "primary": {
    "model": "primary-model-id",
    "context_window": 32768,
    "promote_to": "large",
    "fallbacks": ["backup"]
  },
  "large": {
    "model": "large-model-id",
    "context_window": 131072
  },
  "backup": {
    "model": "backup-model-id",
    "base_url": "https://your-provider.example/v1",
    "api_key_env": ["BACKUP_API_KEY", "BACKUP_API_KEY_SECONDARY"],
    "context_window": 65536
  }
}
```

```powershell
uv run mini_pi.py --route-config routes.json --model primary --max-steps 0
```

上下文晉升先於壓縮，且只會選擇明確設定、視窗更大的 `promote_to`。供應商故障依 `fallbacks` 順序處理。遇到用量限制時，會先嘗試輪換憑證，並優先等待可在時限內解封的備用憑證。`--fallback-on` 可限制備援觸發類別或將其關閉。切換事件會記入 session，重開時若原本的路由設定仍可用，會恢復切換後的模型。[上游上下文晉升](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#context-promotion) · [上游重試策略](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/non-compaction-retry-policy.md)

將路由的 `api` 設為 `anthropic-messages` 可使用原生 Messages 串流。省略 `base_url` 時沿用 `ANTHROPIC_BASE_URL` 或 Anthropic 官方端點。預設讀取 `ANTHROPIC_API_KEY`，也可用 `auth: bearer` 搭配 `ANTHROPIC_AUTH_TOKEN`。可設定 `max_output_tokens` 及 `cache_retention`（`none`、`5m`、`1h`）。Thinking 簽章會保留在 SQLite，且只會在同一路由與相同請求前綴重播。[Anthropic 串流協定](https://platform.claude.com/docs/en/build-with-claude/streaming) · [Anthropic 快取設定](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

若最新的 Anthropic 回合含簽章或遮蔽推理，而且簽發模型仍可重試，備援會略過同供應商的其他模型，跨供應商候選仍可接手。設定 `retry.rateLimitWalksChain: true` 或 `--rate-limit-walks-chain`，可在 HTTP 429 時取消這項暫緩。串流中斷時，已完整收到但尚未執行的工具呼叫會和標示未執行的工具結果一起保留。[上游簽章備援規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/non-compaction-retry-policy.md#operational-caveats)

Anthropic 相容端點若在串流內容開始前拒絕簽章，會先把無簽章的推理改成文字重送，仍失敗時才移除請求裡的推理區塊。這個調整只在目前工作階段的端點和模型生效，SQLite 保留原始推理。已知需要驗證簽章的端點預設使用文字，也可用路由的 `replay_unsigned_thinking` 覆寫。支援 thinking binding beta 的端點可設定 `"thinking_prefix_mismatch": "drop_block"` 交由伺服器處理，或設為 `"error"` 直接顯示前綴錯誤。[上游 transport 實作](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/providers/anthropic.ts#L3398-L3540)

Anthropic 官方 API key 路由會啟用 fallback credit。支援此 beta 的相容路由可設定 `"broker_provider": "anthropic"` 與 `"fallback_credit": true`。收到拒答時，符合上游模型對照、使用同一端點的備援才可兌換。系統提示、工具或對話變更後會重新組請求。兌換 token 和凍結的請求只保留在回合記憶體中，效期五分鐘。[Anthropic fallback credit](https://platform.claude.com/docs/en/build-with-claude/fallback-credit)

在 Oh My Pi 的 profile 啟用 `providers.anthropic.serverSideFallback: true` 後，官方端點上的 Fable 和 Mythos 會要求伺服器在拒答時改用 Opus 5。也可用路由的 `server_side_fallbacks` 指定最多三個模型，例如 `[{"model": "claude-opus-5"}]`，空陣列會停用該路由的鏈。換手後只執行最後模型保留的本機工具呼叫。官方端點且同樣啟用備援鏈時，後續請求才會重播換手標記。[Oh My Pi 設定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/settings-stream-fn.ts#L117-L127) · [Anthropic 伺服器端備援](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#server-side-fallback)

Anthropic 路由可在支援 `compact-2026-09-04` 的部署啟用 `native_compaction: true`。壓縮切點會避開供應商合併後的同角色訊息。只有帶簽章的單一壓縮區塊才會存入 session。重送時保留完整區塊與工具變更，切換到其他路由時則使用區塊中的摘要文字。若原生請求未取得有效區塊，維護會依序嘗試後續方法。 [上游 Anthropic 切點](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/compaction/anthropic.ts#L156) · [Anthropic 原生壓縮](https://platform.claude.com/docs/en/build-with-claude/compaction-on-demand)

Anthropic 快取續期需要啟用 `cache_retention`，並在路由的 `cache_prices` 填入每百萬 token 的美元價格。`input`、`output`、`read` 必填，`write_5m` 和 `write_1h` 可分別覆寫兩種寫入價格。`--cache-warming` 預設為 `idle`，可改為 `streaming` 或 `off`。只有供應商回報快取讀寫、且預期至少節省 0.05 美元時，才會重送原請求的前綴。續期失敗或未命中時會停止，直到下一個正常請求重新確認快取。[上游快取續期](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/settings.md#providers)

路由設為 `api: "openai-responses"` 後，工具結果會以 `function_call_output` 重送。相同路由及前綴會保留 Responses 回傳的 assistant 項目和加密推理。`native_compaction: true` 啟用 V1 `/responses/compact`，再設 `responses_compaction_v2: true` 則優先嘗試串流 `compaction_trigger`。先送完整快照，若供應商回報上下文溢位，則重試切點前的歷史。SQLite、分支複製與 Mini Pi JSONL 都保存快照涵蓋的邊界，回放時只追加邊界之後的訊息。切換到不相容的路由或改用 soft 壓縮時，會從 SQLite 取回原本被壓縮的 transcript。 [上游原生壓縮與前綴備援](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/compaction/compaction.ts#L1845) · [OpenAI Docs 壓縮指南](https://developers.openai.com/api/docs/guides/compaction)

標準 Responses 的 V2 壓縮使用 HTTP/SSE。背景壓縮使用獨立的工作階段識別，手動壓縮沿用目前的識別。V2 若收到上下文溢位，會先用 V2 重試保留尾段之前的歷史，再考慮 V1。`compaction.remoteStreamingV2Enabled: false` 可停用 V2。`compaction.v2RetainedMessageBudget` 控制替換歷史保留的使用者訊息預算，預設及上限都是 64,000 tokens。 [上游 V2 設定與傳輸](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md#summary-generation)

Codex 模型可用 `--model openai-codex/gpt-6-sol` 選擇，憑證來自 `OPENAI_CODEX_OAUTH_TOKEN` 或既有 OAuth 帳號。Codex 預設優先用 WebSocket，可用 `PI_CODEX_WEBSOCKET=0` 或自訂路由的 `prefer_websockets: false` 改走 HTTP/SSE。續接會回放加密推理與完整工具結果，同一回合沿用伺服器的路由識別。完整 Responses 與 Lite 的連線狀態各自保存。同步 V2 壓縮可走 WebSocket，斷線時以 HTTP 重送完整壓縮請求。背景 V2 壓縮使用獨立識別碼並走 HTTP/SSE。 [上游 Codex Responses](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/providers/openai-codex-responses.ts)

開啟 `providers.openaiLiveSteering: true` 後，終端 Enter 或 RPC `steer` 可將新指令送進支援 steering 的 Codex WebSocket 回應。伺服器接受後，程式會讀取自動開始的下一段回應，或等工具完成後只傳工具結果。拒絕或逾時的指令會回到佇列，等下一個可插入的邊界。此設定預設關閉。 [上游 steering 協定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/providers/openai-codex/live-steering.ts)

支援 `configuration_update` 的標準推理模型會沿用第一次送出的 `reasoning.effort`。切換強度時，Responses 和 Codex 會在新使用者訊息前或工具結果後加入更新，重送時保留原順序與加密推理。原生壓縮請求略過這些更新，工作階段重置後重新建立基準。相容代理可在模型設定中指定 `compat.supportsConfigurationUpdate: false`，直接使用請求層級的強度。 [OpenAI 推理強度切換](https://developers.openai.com/api/docs/guides/reasoning#change-reasoning-mid-conversation) · [上游基準管理](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/providers/openai-configuration-update.ts)

Apple Silicon macOS 上，正式 Codex OAuth 的 HTTP 請求和 WebSocket 升級會即時準備 DeviceCheck 證明。在圖形登入工作階段使用 Apple API 取得 token，並等待回呼最多一秒。其他平台和自訂 Codex 端點略過該標頭。在 macOS 手動安裝 Python 套件時，可安裝 `pyobjc-framework-DeviceCheck` 啟用此功能。 [上游證明格式](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/live/attestation.ts) · [Apple DeviceCheck](https://developer.apple.com/documentation/devicecheck/dcdevice)

視覺路由可設定 `vision: true`，讓 Snapcompact 在本機把舊歷史排成依序重放的文字和 PNG。原文與圖框一併保存在 SQLite，再次壓縮時會重新排版。切換到文字路由或供應商拒絕圖像時，會改用同一份封存文字。`--snapcompact-shape` 預設 `auto`，也可指定版型。路由可用 `image_max_edge`、`image_count_budget`、`image_byte_budget`、`image_tokens` 和 `image_detail` 配合實際部署。字型採上游資產，Silver 的授權與作者資訊隨附。[上游 Snapcompact](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md#snapcompact-method)

`--snapcompact-system-prompt agents-md` 可在估算節省 token 時將已載入的專案指令轉成圖框，`all` 則可處理整份系統提示。預設 `none`。專案指令支援 `AGENTS.md`、`CLAUDE.md`、`.omp/`、`.agent/`、`.agents/` 及工作目錄的 Claude、Gemini、Copilot 指令檔，同一層只取優先順序最高的檔案。`--snapcompact-tool-results` 允許成像大型成功工具結果。讀檔預設保留文字，支援的 Opus 路由會啟用讀檔成像，也可以在路由用 `image_read_results` 覆寫。成像只改寫當次請求。圖片超過額度時會撤回較新的成像結果並送出原文。[上游即時成像規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/snapcompact-inline.ts)

互動模式的 `/handoff [交接重點]` 會用當前模型、對話前綴及工具清單生成交接文件，並在同一個 session 內保留近期批次。交接請求先停用工具選擇。只接受自動選擇的供應商會改用 `auto` 重試，交接只採用回傳的文字。自動壓縮可事先在背景生成交接文件。上下文溢位時可套用已生成且仍有效的結果。`--handoff-save-to-disk` 會將自動生成的文件另存到該 session 的 artifacts 目錄。[上游 handoff 流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/handoff-generation-pipeline.md)

標準 Responses 路由可用 `providers.openaiResponsesWebsockets: true` 對已標示支援的模型啟用 WebSocket，或在自訂路由設 `prefer_websockets: true`。同一條連線只在設定與歷史前綴吻合時傳送差量。續接 ID 失效會重送完整輸入，連續三次失效會停止傳續接 ID。端點拒絕 WebSocket 升級時會沿用 HTTP/SSE，直到工作階段刷新。`/fresh`、`/clear` 與上下文重寫會關閉舊連線。 [上游標準 Responses WebSocket](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/models.md#standard-responses-websockets) · [OpenAI WebSocket 文件](https://developers.openai.com/api/docs/guides/websocket-mode)

標準 OpenAI Responses 的 HTTP/SSE 在正式 OpenAI 端點預設啟用續接，送出 `store: true` 讓供應商儲存回應。工作階段、帳號、請求設定及歷史吻合時，透過 `previous_response_id` 傳送新增項目。HTTP 與 WebSocket 各自保存續接狀態。過期 ID 會以完整歷史重試一次，連續三次遭拒後停用續接。遇到零資料保留拒絕會立即以 `store: false` 重試。可在自訂路由設 `stateful_responses: false`，或用 `PI_OPENAI_STATEFUL=0` 關閉續接。自訂端點預設依模型相容性設定決定，未宣告支援時採完整重送。 [上游 HTTP 續接](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/providers/openai-responses.ts) · [OpenAI 工作階段文件](https://developers.openai.com/api/docs/guides/conversation-state#passing-context-from-the-previous-response)

標準 Responses 的快取保留可在路由用 `cache_retention` 設定，依序優先於 `providers.cacheRetention` 和 `PI_CACHE_RETENTION`。預設 `short`。`long` 在新版相容模型送出 12 小時快取選項，在支援舊版保留欄位的模型送出 `24h`。`none` 省略快取鍵及快取選項，並停用 HTTP 差量續接。支援明確快取的模型可用 `"prompt_cache": {"mode": "explicit", "ttl": "30m"}` 在最新輸入前的穩定內容標記斷點。HTTP 追加時沿用既有斷點，過期 ID 的完整重試會重新選擇斷點。可設 `"breakpoint": "none"` 省略標記，或用 `"mode": "implicit"` 交由供應商選擇。 [上游快取選項與斷點](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/test/openai-responses-cache-affinity.test.ts)

Muse Code 可設定 `providers.museCode.storeResponses: true`，或用 `PI_MUSE_STORE_RESPONSES=1` 啟用供應商儲存回應。SSE 斷線時，若已有回應 ID 且只收到推理內容，會用同一個端點和憑證每五秒查詢一次原回應，最多兩分鐘或 24 次。取回的文字、工具呼叫與用量交回原本的回合處理，已串流的推理不會重播。收到可見答案或工具草稿後斷線時，沿用一般的串流錯誤處理。 [上游斷線取回](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/test/openai-responses-resume.test.ts) · [Responses 取回 API](https://developers.openai.com/api/reference/resources/responses/methods/retrieve)

## 已移植的行為

- 設定 `compaction.experimentalContextManagement: true` 可使用實驗性的筆記型上下文視窗。`context_notes` 讀取或整份替換目前分支的筆記，空字串會清空，單份上限為 16,384 UTF-8 位元組。`new_context` 在工具批次結束後換窗，即使關閉自動壓縮也能使用。達到門檻或使用無額外指示的 `/compact` 也可換窗。換窗保留筆記、最近的完整工具批次與需要接續的使用者請求，並保留原始歷史。 [上游筆記型上下文視窗](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md#experimental-notes-backed-context-windows)
- 啟用後，`read` 和 `grep` 可透過 `history://current/full` 讀取或搜尋目前分支的原始歷史，包含工具結果和換窗界線。`read` 支援 `:1-200` 與 `:raw:1-200`。筆記與換窗記錄會隨工作階段日誌匯出，`/clear` 會隱藏較早的筆記修訂。 [上游歷史存取](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/tools/context-notes.md)
- 設定 `--context-window` 後，預設在估算用量接近 70% 時準備背景摘要。若單次回合跳過門檻，只要仍在寬限區間內，前景便繼續執行。碰到區間上限或收到上下文溢位錯誤時，尚未完成的背景請求會取消，交由同步維護處理。追加的訊息與完整的工具呼叫和結果都會保留。若追加訊息使背景摘要無法騰出足夠空間，會用目前歷史重新摘要。`compaction.asyncEnabled: false` 可停用背景摘要。第一個可用方法為 Shake 或 Snapcompact 時會直接走同步維護。 [上游背景壓縮流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/session-maintenance.ts#L2329)
- 送出新回合前，排隊提示會原樣保留，並計入壓縮後的剩餘空間。若已就緒的背景摘要在跨過門檻時失效，當次維護會重新選擇模型或立即壓縮。 [上游背景摘要的提交條件](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/session-maintenance.ts#L2476)
- 本機門檻估算計入系統提示、模型收到的工具 schema、按尺寸計價的圖片，以及可重播的 Responses 歷史。密文長度不計入。分詞依 catalog 選擇，包含四種 Claude 家族和 `cl100k`、`o200k`、Qwen、DeepSeek、Kimi、GLM。缺少家族資料時使用 `o200k`。詞表和 Claude 正規化程式的授權收錄於 `catalog/LICENSE.*`。Responses 原生壓縮會先做視窗檢查，必要時改送較短前綴或縮減請求尾端的工具輸出。 [上游分詞表來源](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/crates/pi-natives/data/README.md) · [ctok 正規化程式](https://github.com/sanderland/ctok/tree/df3b59b5e645289a5eadc8e24036b99d39c333c4/ctok) · [上游原生壓縮視窗檢查](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/compaction/openai.ts#L253)
- `compaction.midTurnEnabled: false` 可停用主代理工具迴圈中的門檻檢查。開跑前與回合結束時仍會檢查，子代理在執行中仍會檢查。門檻判斷同時採用本機完整歷史估算和最近一輪的供應商上下文用量。Responses 的 orchestration token 從上下文估算扣除，計費紀錄保留原始數字。背景摘要仍在執行且用量仍在寬限區間時，前景可繼續用原模型。摘要就緒或超過寬限區間時，才進入正式維護或切換大視窗模型。若某個工具迴圈切點無法壓縮，會等切點前移再試。 [上游回合中的維護流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/session-maintenance.ts#L2841)
- 回合結束後的門檻壓縮會確認剩餘上下文已降至壓縮門檻的 80% 以下。有足夠空間且仍需接續時，代理會加入一則續跑訊息，活動目標也可沿這條路徑接續。已排隊的使用者訊息優先，普通最終文字答覆會收尾。`compaction.autoContinue: false` 可停用這則壓縮後的接續訊息。若仍無足夠空間，會暫停自動接續並回報警告。 [上游壓縮接續](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/agent-session.ts#L5603)
- 壓縮卡住時，救援可在縮小 Snapcompact 快照和卸載大型內容後，從模型上下文移除附件圖片。`/shake images` 可手動執行這一步。文字、工具配對和原始分支歷史會保留。供應商拒絕圖片時可用清理後的上下文重試。壓縮前後的估算用量和空間不足警告會隨壓縮紀錄保存，JSONL 搬移和 RPC 的會話訊息都可查看。 [上游圖片救援與警告](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/session-maintenance.ts#L6275)
- `compaction.thresholdTokens` 的正值優先於 `compaction.thresholdPercent`。百分比可設為 `-1`，以視窗扣除預留空間作為門檻。固定門檻達到或超過實際視窗時，也改用這個預算。用 `compaction.reserveTokens` 設定預留量，或用 `compaction.keepRecentTokens` 調整最多保留多少近期歷史。`--compact-at` 與 `--keep-recent` 可覆寫百分比與近期保留量。長度截斷時會分別檢查提示是否跨過壓縮門檻，以及整輪用量是否跨過視窗預算。 [上游門檻與預留計算](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/compaction/compaction.ts#L325)
- `compaction.modelThresholds` 用 `provider/model` 比對，找不到完全相符的鍵時，選最長的 `provider/prefix*`。例如數字 `90000` 是套用百分比或預留規則的基準，字串 `"f90000"` 是固定門檻，`"80%"` 則依視窗計算。若 token 門檻需要更大視窗且目錄有該規格，會啟用它。`task.agentCompactionThresholdOverrides` 以代理名稱比對，接受固定 token 數字或百分比，優先於模型設定。專案層可用 `null` 清除繼承的單一項目。 [上游覆寫規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/config/compaction-threshold.ts#L52)
- `compaction.supersedeReads` 和 `compaction.dropUseless` 預設開啟。回合完成後，完整的檔案重讀可取代舊的範圍讀取，raw 形式則各自比對。同一工具呼叫及其結果位置會保留，替換只影響該分支的模型上下文。修剪會顧及約 8,000 token 的尾端預算及 Anthropic 快取回看範圍。工作階段閒置達 90 分鐘後可在下一次送出前清理更深的過時結果。 [上游修剪規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/compaction/pruning.ts)
- 壓縮前的大量工具結果修剪會保護最近 40,000 個工具輸出 token，並要求至少可節省 20,000 個 token。無效益標記的成功工具結果及其配對呼叫會從本機摘要與 Snapcompact 的輸入排除。原始工具結果保留在日誌。 [上游預壓縮修剪](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md#pre-compaction-pruning)
- `--compaction-methods` 預設依序使用 `native,snapcompact,handoff,shake,soft`。原生壓縮與 Snapcompact 各依路由能力啟用。`shake` 會把較舊的工具輸出及大型 fenced code block／XML 區塊改成可讀回的引用，並保留約 16,000 token 的近期上下文。手動 `/shake` 保護約 4,000 token。若上下文溢位且一般維護無法繼續，救援路徑可處理最新的大型結果。引用和原文在同一筆交易寫入，原始 transcript 與工具呼叫 ID 均保留。[上游 Shake 規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md#shake-method)
- OpenAI Responses 完整 HTTP 請求收到 `408 Timed out reading request body` 時，只有在 `compaction.enabled` 開啟、方法清單包含 `shake`、仍有重試額度且舊工具結果可以節省至少 4,000 個估算 token 時，才會儲存原文並用同一模型重送一次。最近約 16,000 個 token、工具圖片、原生 Responses 歷史和其他訊息會保留。模型產生新輸出後，之後的完整請求可以再次觸發恢復。帶 `previous_response_id` 的增量請求會保留原本的 ID 和內容重試。可在 `compaction.methodOrder` 設定方法清單，其中 `remote` 對應 CLI 的 `native`。 [上游 Responses 408 恢復規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/non-compaction-retry-policy.md#responses-request-body-read-timeout-exception)
- 開啟 `--memory local` 後，啟動時在背景找同一工作目錄下、已完成且符合閒置門檻的 session。第一個模型階段抽取可重用資訊，第二個階段合併成後續 session 的指引。預設閒置門檻是 12 小時。`--maintenance-model` 可替背景請求指定另一個模型。
- Session 按工具批次持久化。執行中斷時保留配對的工具結果位置，續接時不會重新執行已發出但未留下結果的工具。
- 串流斷線時，已顯示且沒有工具草稿的文字會和續寫提示一起存入 session。含有未完成工具呼叫的請求會從既有歷史重新送出。暫時錯誤預設最多同路由重試 10 次，`--retry-max-delay` 預設將等待上限設為 300 秒。超過輸出上限時，低於壓縮門檻的可見文字會保留，上下文溢位則先嘗試晉升，再進入壓縮恢復。[上游恢復流程](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/turn-recovery.ts)
- Responses 或 Codex 遇到輸出 token 上限時，agent 會檢查整批函式呼叫是否都有完整參數的串流證據。可執行的批次沿用正常工具派送。其餘可識別的函式呼叫會儲存同一批次的未執行結果，讓模型接著把工作切成較小的呼叫。 [上游終止事件測試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/test/openai-responses-stream-terminal.test.ts)
- Responses 或 Codex 回傳 `end_turn: false` 時會保存進度並繼續採樣。標準 Responses 完成供應商端網路搜尋但尚無可見答覆時也會接續，下一個請求會重播原生搜尋項目。連續自動接續上限是八次，出現本機工具批次後會重新計數。若先遇到模型步數上限，可用 `/continue` 從已存進度接續。 [上游回合暫停測試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/test/agent-loop.test.ts)
- DeepSeek 的 Chat Completions 回應若把完整 DSML 或舊版模板呼叫寫成文字，會轉成一般工具呼叫。同輪已有結構化工具呼叫時，以原生呼叫為準。殘留的破損標記會在寫入歷史前清除，並最多提示模型修正兩次。程式碼區塊與行內範例會保留原文。 [上游 DeepSeek 串流修復](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/test/stream-markup-healing.test.ts)
- Chat Completions 收到 `[DONE]` 就能確認回應完整，即使端點省略 `finish_reason`。已有輸出但兩種結束訊號都缺少時會走中斷恢復。收到 `finish_reason` 後最多等 2.5 秒收取尾段用量。用量齊全時先完成前景回合，再由原連線在背景收取最後的 `[DONE]`。 [上游串流終止測試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/test/openai-stream-terminal-close.test.ts)
- OpenAI 家族的 HTTP 串流分開計算回應標頭、首個有效事件與後續事件的等待時間。心跳和空事件不會延長等待。可用 `PI_OPENAI_STREAM_FIRST_EVENT_TIMEOUT_MS` 和 `PI_OPENAI_STREAM_IDLE_TIMEOUT_MS` 調整，設為 `0` 可關閉對應的語意逾時。Codex WebSocket 使用 `PI_CODEX_WEBSOCKET_FIRST_EVENT_TIMEOUT_MS` 和 `PI_CODEX_WEBSOCKET_IDLE_TIMEOUT_MS`。各項預設為五分鐘，模型目錄可以覆寫 HTTP 預設。 [上游首事件逾時測試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/test/openai-first-event-timeout.test.ts)
- Chat Completions 的 `finish_reason` 或 `stop_reason` 回報重複生成時，agent 會捨棄失敗的推理和工具草稿，依 `--retry-max-retries` 在同一模型重試。啟用 loop guard 時，每次重試會加入固定的隱藏修正提示。失敗請求已回報的用量仍會入帳。 [上游 repetition stop 測試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/test/issue-12525-repro.test.ts)
- 帳號配額耗盡時，先嘗試切換可用帳號或兌換額度。需要等待時，會比較目前帳號與適用備用帳號的解封時間，再用可用帳號重試同一回合。可用 `--wait-for-usage-reset` 或 profile 的 `retry.waitForUsageReset: true` 允許供應商提供的重置時間超過一般等待上限。CLI 的 `--retry-max-delay` 優先於 profile 的 `retry.maxDelayMs`。暫時限流仍受一般上限約束，餘額不足會回報原錯誤。配額等待採牆上時鐘，機器休眠時間也計入，並可在 CLI 用 Ctrl+C 或在 RPC／ACP 取消。[上游等待規則](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/turn-recovery.ts#L2801-L2969)
- 自動摘要、handoff 和原生壓縮有獨立的重試預算。仍有可用候選時，超過 30 秒的等待會改用下一個模型。只有最後一個候選能依 `retry.waitForUsageReset` 等過一般上限。候選從已設定的路由、可對應路由的 profile 模型角色及最大視窗備選選出，且不會切換前景回合的模型。取消背景壓縮會中斷等待和進行中的 HTTP 請求。[上游自動壓縮重試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/session-maintenance.ts#L5849-L5953)
- Thinking Loop Guard 在串流中檢查尾端精確重複、相似段落及詞彙停滯。偵測後會關閉當次串流並重新請求，合計最多三次。`--loop-guard` 可以明確選擇 `off`、`exact` 或 `semantic`，預設 `auto` 依模型名稱判斷。工具只在收到完整回應後執行。[上游偵測器](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/utils/thinking-loop.ts)

`/tree` 列出目前 session 的節點與分支。用 `/tree <節點編號>` 原地切換，或用 `/tree <節點編號> --summary [重點]` 摘要離開的分支後切換。選取使用者節點會回到它的父節點並填入原提示。可用 `/label <節點編號> [名稱]` 標記節點。`/fork` 複製整個 session 並切換到新 UUID，`/fork <節點編號>` 只複製該節點的祖先路徑。也可以在啟動時用 `--fork <來源 UUID>` 複製本地 session。[上游 session tree](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/tree.md)

用 `--import-jsonl <檔案>` 或 `--fork <JSONL 路徑>` 建立新 session，可加 `--continue` 續接未完的回合。用 `--session <UUID> --export-jsonl <新檔案>` 匯出 v3 日誌，副檔名 `.gz` 會壓縮日誌。有 artifact 時會另寫同名目錄，例如 `work.jsonl.gz` 搭配 `work/`。搬移時請一起保留兩者，匯入後 `read` 會沿用原本的 `artifact://` ID，包含 Oh My Pi 的數字 ID。舊版 pipy 日誌中的 Shake 引用也可從完整原文按雜湊復原。匯入保留分支、工具結果、標籤與重置邊界。外部 blob 仍由原來的儲存位置管理。 [上游 session 格式](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/session.md#session-file-format) · [上游 fork 行為](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/session-operations-export-share-fork-resume.md)

互動終端可用 `--restore <UUID>` 接續前一個程序中斷的回合，包含來不及寫下退出紀錄的情況。工作階段會分別記錄工具開始、工具結果與程序退出。若整批工具都失敗或沒有留下結果，且日誌保留可執行的呼叫與參數，會直接重跑該批。已留下成功結果的批次會保留既有結果，並交由模型從中斷處繼續。 [上游中斷恢復](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/agent-session.ts#L12936)

啟用背景工作時，`--restore` 會沿用中斷子代理的工作 ID 和工作階段，先登記工作，再讓各層父代理接續。父代理原本的 `wait` 可以等到恢復後的結果。中間父代理已結束時，仍會恢復它底下符合條件的工作，並把通知保存給原本的擁有者。通知寫進該工作階段日誌後才會標為已接收。已結束、明確取消或使用隔離工作目錄的子代理會略過。`workpool` 批次仍待移植。 [上游子代理恢復](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/task/restore.ts)

互動終端、RPC 和 ACP 可用 `/retry` 重試上一個失敗或中止的回合。失敗輸出保留在日誌中，下一次模型請求會略過它。若上一批工具的每個結果都失敗，且呼叫參數完整、之後沒有需要保留的背景通知，代理會直接重跑該批。舊版 pipy 日誌也支援這項重跑，原本的失敗結果仍保留在舊分支。 [上游手動重試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/turn-recovery.ts#L3429)

使用者主動中斷串流時，原始推理留在工作階段日誌。尾端未簽章的推理達到 60 字元時，下一個回合可以透過隱藏提示接續。已完成的原生推理區塊由各供應商的重播規則處理。Claude 等會拒絕引述推理的模型會略過這類提示，切換到這些模型後也會略過。 [上游中斷推理測試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/test/agent-session-interrupted-thinking.test.ts)

跨機器可用 `session pack <UUID 或 JSONL> <archive.tar>` 和 `session unpack <archive.tar> --cwd <工作目錄>`。tar 使用 Oh My Pi 的 v1 格式，包含日誌、artifact 目錄及日誌引用的 blob。打包原版日誌時可用 `--blobs-dir` 指定上游 blob 目錄。解包前會驗證成員路徑與 SHA-256。如果本機已有較早且相容的歷史，會沿用工作階段 ID 更新。若本機歷史已分歧，會建立分支並保留原工作階段。`unpack` 在 stdout 輸出可交給 `--session` 的 UUID。 [上游 session transfer](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/session-transfer.ts)

用 `session moved <UUID 或 JSONL> --to <目的地名稱>` 標記留在原機的副本。再次開啟該副本時，pipy 會從新 UUID 續跑，保留原來的工作階段和工具結果，並在分支中補記缺失的工具結果。`session lease --json -- <ID>...` 可查詢 Oh My Pi 相容的所有權鎖。已有程序持鎖時，同 ID 的打包、標記和更新會回報工作階段忙碌。 [上游搬移標記與續跑防護](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/session-transfer.ts#L651)

匯入 Oh My Pi 的 OpenAI Responses 原生壓縮時，使用同一供應商的 Responses 路由會重播快照，再接上快照建立後新增的訊息。切到其他路由時會從可讀的摘要與逐字稿重建上下文，切回相容路由時可再次使用快照。重匯出也會保留快照和重播邊界。 [上游壓縮紀錄格式](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md#session-entry-model)

Anthropic Messages 的原生壓縮匯入會保留簽章、舊版加密狀態和工具變更，並按原時間點插入檔案紀錄。相容路由可重播保留回合的推理區塊，其他路由會讀取文字摘要。舊版 pipy 已匯入的工作階段需要從原 JSONL 重新匯入，才能取回先前未儲存的簽章。 [上游 Anthropic 壓縮](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/agent/src/compaction/anthropic.ts)

Bedrock 路由可選 `amazon-bedrock/<模型 ID>`，也可在路由設定中使用 `bedrock-converse-stream`。認證可用 `AWS_BEARER_TOKEN_BEDROCK` 或標準 AWS 憑證鏈。一般 Converse 回合使用可讀摘要，支援原生壓縮的模型會在 InvokeModelWithResponseStream 壓縮時重播同供應商的簽章與工具變更。 [上游 Bedrock 路由](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/providers/amazon-bedrock.ts) · [AWS 壓縮介面](https://docs.aws.amazon.com/bedrock/latest/userguide/claude-messages-compaction.html)

`/fresh` 會關閉已快取的模型 HTTP 連線、清除帳號釘選並換一個暫時的供應商識別碼，同一份工作階段逐字稿會繼續用於模型請求。JSONL RPC 可用 `fresh_session` 取得本次刷新資訊。 [上游供應商工作階段刷新](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/session-operations-export-share-fork-resume.md)

`/clear` 在同一工作階段新增重置邊界，保留工作階段 ID、標題和完整逐字稿。模型回放、原生壓縮狀態與自動回憶查詢從新邊界重新開始。Hindsight 和 Mnemopi 會用新的逐字稿識別碼保存後續對話。互動終端會清除畫面與捲動區，仍可透過 `/tree` 回到較早的分支。JSONL 匯出包含重置前的歷史與 `reset_boundary`。CLI 與 ACP 支援 `/clear`，JSONL RPC 也可直接呼叫 `reset_context`。 [上游工作階段重置](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/session/agent-session.ts#L6864)

## JSONL RPC

```powershell
uv run mini_pi.py --mode rpc --route-config routes.json --model primary --cwd 'E:\your-repo'
```

宿主使用 UTF-8，一行送一個 JSON 指令。stdout 專供 RPC，診斷寫到 stderr。收到 `ready` 後可以送 `{"type":"prompt","id":"p1","message":"請檢查專案結構"}`。同一個 `id` 會先出現在 `response`，完成後再出現在 `prompt_result`。等到 `session_settled` 表示本輪和已排隊的輸入都已處理。可用 `negotiate_protocol` 選版本 2，讓超過 1 MiB 的回應拆成可重組的 `rpc_chunk`。

宿主可以收串流文字、工具事件與維護事件，也可以在串流時送 `steer`、`follow_up`、`abort`、`abort_and_prompt` 或 `abort_and_restore_queue`。取消可中斷等待 HTTP 回應標頭、讀取串流及本機 shell 的執行。短暫的同步檔案操作會在下一個檢查點交回控制。

歷史可用 `get_entries`、`get_tree`、`get_messages_page` 讀取。空閒時支援 `navigate_tree`、`fork`、`new_session`、`switch_session`、`compact` 和 `handoff`。`switch_session` 接受本機 SQLite session 的 UUID 或可匯入的 JSONL 路徑。`get_available_models` 也列出啟動時已載入的探索模型。`set_model` 可用路由別名，或直接傳 `provider` 和 `modelId`。[上游 RPC 協定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/rpc.md)

## Agent Client Protocol

```powershell
uv run mini_pi.py --mode acp --route-config routes.json --model primary
```

把這組指令設為 ACP 編輯器啟動的 agent 程序。工作目錄由每個 `session/new` 或 `session/load` 請求指定。同一個程序可以同時處理多個 SQLite session。`session/load` 重播作用中歷史，`session/resume` 直接接續。也可列舉、fork 或關閉 session。編輯器可用 `session/cancel` 停止回合，或用 `_session/steering` 注入當前回合。`/compact`、`/handoff` 和 `/continue` 可從 ACP prompt 呼叫。[ACP v1 協定](https://agentclientprotocol.com/protocol/v1/overview)

如果編輯器在 `initialize` 宣告了檔案或終端能力，對應工具會使用編輯器端的讀寫與終端。Shell 依 ACP 權限回覆執行，取消時會關閉等待中的權限對話。模型和推理設定以本機路由為準。ACP 目前支援預設模式與既有憑證，計畫審批和額外登入流程仍待移植。

## MCP 工具

ACP 依 `session/new`、`session/load`、`session/resume` 或 `session/fork` 提供的 `mcpServers` 連線。CLI 和 JSONL RPC 會讀取專案 `.omp/`、Oh My Pi 使用者目錄與專案根目錄的 `mcp.json` 或 `.mcp.json`。可以用 `--mcp-config <路徑>` 加入高優先順序的設定，或用 `--no-mcp` 停用磁碟探索。

支援 stdio、Streamable HTTP 和 SSE。服務可並行啟動，較慢的服務會在背景繼續連線。CLI 和 RPC 會快取工具定義，讓模型可以選用尚在連線的已知工具，執行時再等服務就緒。新的工具清單在下一次模型請求生效。取消會傳到執行中的 MCP 工具，session 關閉或換設定時會清理連線與子程序。

用 `/mcp list` 查狀態，`/mcp reconnect [名稱]` 重新連線。互動 CLI 另支援 `/mcp reload`。遠端驗證可由設定提供標頭。其他工具的設定檔匯入仍待移植。[上游 MCP 生命週期](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/mcp-runtime-lifecycle.md) · [MCP Python SDK](https://py.sdk.modelcontextprotocol.io/client/)

遠端伺服器可用 `/mcp reauth <名稱>` 登入。用瀏覽器開啟輸出的連結，透過本機 loopback 完成 PKCE 回呼。預設回呼路徑為 `/callback`，優先使用連接埠 3000。RPC 宿主透過 `notice` 取得連結，ACP 宿主透過 `session/update` 取得連結。支援 OAuth metadata 探索、動態 client 註冊與設定中既有的 `oauth.clientId`。設定 `auth.type` 為 `apikey` 時仍由設定的標頭提供 API key。顯式 `Authorization` 標頭優先，除非 `auth.credentialId` 可解析為本 profile 的受管理憑證。[上游 MCP 設定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/mcp-config.md#auth-fields) · [MCP 授權規格](https://modelcontextprotocol.io/specification/2025-11-25/basic/authorization)

伺服器即使接受匿名初始化，`/mcp reauth` 仍會查詢它發布的 OAuth metadata。HTTP 401 與工具授權挑戰也會讀取 `resource_metadata`、`Mcp-Auth-Server`，或舊式標頭與 JSON 內的登入端點。探索會優先查資源專屬文件，並依它公布的 issuer 清單尋找 OAuth 或 OpenID Connect metadata。也支援 `/oauth/metadata` 與 `/.mcp/auth`。若服務要求特定 scope，動態註冊與登入會帶入該 scope。找不到可用的授權資料時會結束指令。[上游授權探索](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/mcp/oauth-reauth.ts#L398-L447)

權杖預設存於 `--state-dir` 下的 `mcp-oauth.sqlite3`，依 `OMP_PROFILE` 或 `PI_PROFILE` 和完整伺服器 URL 隔離。Windows 使用帳號的 DPAPI 保護資料。其他平台以 `0600` 權限建立憑證檔。請求送出前會檢查效期，收到 HTTP 401 時會嘗試刷新並重送一次。共用同一筆憑證的程序會序列化刷新，避免重複使用已輪替的 refresh token。`/mcp unauth <名稱>` 移除對應憑證。

MCP 和指定 `broker_provider` 的模型路由可使用 Oh My Pi 或 Python 的 Auth Broker。設定 `OMP_AUTH_BROKER_URL` 和 `OMP_AUTH_BROKER_TOKEN`，或在 profile 的 Oh My Pi `config.yml` 設定 `auth.broker.url` 和 `auth.broker.token`。也可從 profile 的 `auth-broker.token` 讀取 bearer。MCP 登入寫入 broker、登出停用 broker 憑證、refresh 由 broker 執行。[上游 Auth Broker 協定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/auth-broker-gateway.md)

在 Mini Pi 目錄執行 `uv run auth_broker_server.py --agent-db PATH` 可啟動 Python broker。預設監聽 `127.0.0.1:8765`。省略路徑時使用目前 profile 的 `agent.db`，缺檔時建立 v8 憑證庫。服務讀取 `OMP_AUTH_BROKER_TOKEN` 或 profile 的 `auth-broker.token`，兩者皆未提供時建立 token 檔。

服務支援快照、SSE、條件長輪詢、憑證寫入與停用、作用域封鎖及用量回報。快照遮蔽 OAuth refresh token 和 MCP client secret。Anthropic、Codex 與受管理 MCP 憑證共用 SQLite 續期租約，背景每分鐘檢查一次，可用 `--no-refresh` 停用背景檢查。[上游服務端](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/ai/src/auth-broker/server.ts)

Python broker 和本機 `agent.db` 路由會查詢 Anthropic 與 Codex 的配額端點。每個帳號的成功報告在 SQLite 快取約五分鐘，並使用隨機錯開的過期時間。同一帳號的並行輪詢會共用請求。暫時失敗時，24 小時內的最近成功報告可供選帳使用，並保留原本的時間戳。Codex 的授權拒絕會清除舊報告。手動使快取失效後，同供應商的下一輪查詢依序探測各帳號。可用的 Claude 與 Codex 重置額度也會顯示在報告中。其他供應商的即時配額與 OAuth 續期、舊版資料庫遷移與 Auth Gateway 仍待移植。[上游配額快取](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/auth-broker-gateway.md#usage-cache-server-side-5-min-jitter--client-side-15-s-single-flight)

Python 與 Oh My Pi 可互讀 AES-256-GCM 加密的 v2 broker 快照。預設在 profile 的 `cache/auth-broker-snapshot.enc`，也可用 `OMP_AUTH_BROKER_SNAPSHOT_CACHE` 指定路徑。快照從 broker 產生時開始計算 TTL，預設一小時。`OMP_AUTH_BROKER_SNAPSHOT_TTL_MS=0` 停用讀寫。啟動時會先載入有效快照，並在背景透過 SSE 或 ETag 長輪詢同步。權杖過期時仍須由 broker 刷新。[上游快照格式與設定](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/auth-broker-gateway.md#client-snapshot-cache)

工具若透過錯誤結果的 `_meta["mcp/www_authenticate"]` 要求授權，當輪會等待使用者開啟連結。登入完成後會重連，並在原本的工具呼叫內重試一次。取消該輪也會中斷登入並關閉回呼監聽。RPC 的登入連結以帶有 `actionRequired: true` 的 `notice` 傳送，即使宿主已過濾一般通知。[上游 MCP 工具授權重試](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/packages/coding-agent/src/mcp/tool-bridge.ts#L403-L440)

`/mcp resources [伺服器]` 列出資源與模板。用 `read` 傳入原生 URI 或 `mcp://<uri>`，會先找完全相符的資源，再挑最具體的模板。`/mcp prompts` 列出 `/伺服器:提示名稱` 命令。可以傳入 `key="帶空格的值"`，展開後由現有 agent loop 處理。ACP 和 JSONL RPC 會在命令清單變動時通知宿主。[MCP 資源](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) · [MCP 提示](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts)

用 `--mcp-notifications` 或 `/mcp notifications on` 訂閱資源更新。同一資源的短時間連續通知會合併，預設等待 500 毫秒。通知會在模型請求邊界交付，或在 ACP、JSONL RPC 和互動終端空閒時啟動背景回合。宿主取消回合後，更新會留待下一次明確送出的 prompt。透過管線輸入的 CLI 會在下一個 prompt 交付更新。

目前快取續期支援 Anthropic Messages。其他記憶後端、外部 artifact／blob 互通、完整的樹狀選擇器與指令檔探索、RPC 宿主擴充、其餘 ACP 擴充和完整工具集仍需移植。[上游壓縮行為](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/compaction.md) · [上游記憶行為](https://github.com/can1357/oh-my-pi/blob/24439066040e006eb08da7b204f0ac41c79d2de9/docs/memory.md)
