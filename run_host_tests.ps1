param(
    [string]$LogPath = ".\test_after_cleanup.log"
)

$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $repoRoot

function Write-Log {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string]$Message
    )

    Write-Host $Message
    Add-Content -LiteralPath $script:ResolvedLogPath -Value $Message -Encoding UTF8
}

function Invoke-LoggedCommand {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Command
    )

    # 直接捕获 stdout/stderr 并逐行写入日志，避免 Start-Transcript 漏掉子进程输出。
    Invoke-Expression "$Command 2>&1" | ForEach-Object {
        Write-Log ([string]$_)
    }
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "命令执行失败，退出码: $exitCode"
    }
}

function Invoke-TestStep {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Title,
        [Parameter(Mandatory = $true)]
        [string]$Command,
        [string]$Prompt = ""
    )

    Write-Log ""
    Write-Log "============================================================"
    Write-Log "TEST: $Title"
    Write-Log "COMMAND: $Command"
    if ($Prompt) {
        Write-Log "PROMPT:"
        Write-Log $Prompt
    }
    Write-Log "============================================================"
    Invoke-LoggedCommand $Command
}

$script:ResolvedLogPath = Join-Path $repoRoot $LogPath
"===== Host voice test started: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" | Add-Content -LiteralPath $script:ResolvedLogPath -Encoding UTF8
try {
    Invoke-TestStep `
        -Title "1. 静态检查" `
        -Command "python -m py_compile main_cli.py wake_loop.py streaming_voice.py streaming_tts.py audio_session.py prompt_audio_cache.py llm_client.py audio_output.py settings.py"

    Invoke-TestStep `
        -Title "2. 单轮唤醒对话" `
        -Command "python main_cli.py --wake-loop --wake-rounds 1" `
        -Prompt "请依次说：你好；介绍一下成都周边的景点"

    Invoke-TestStep `
        -Title "3. 连续 3 轮上下文测试" `
        -Command "python main_cli.py --wake-loop --wake-rounds 3" `
        -Prompt "请依次说：你好；介绍一下成都比较适合下雨天去的室内景点；那有没有具体的室内游乐场；那室外有没有适合带老人去的公园"

    Invoke-TestStep `
        -Title "4. 停顿防截断测试" `
        -Command "python main_cli.py --wake-loop --wake-rounds 1 --wake-end-silence-seconds 1.0" `
        -Prompt "请依次说：你好；我想问一下……停顿 1 到 2 秒……成都适合老人去的公园有哪些"

    Invoke-TestStep `
        -Title "5. 空输入回待机测试" `
        -Command "python main_cli.py --wake-loop --wake-rounds 1 --exit-after-idle-return" `
        -Prompt "请先说：你好；听到确认后保持安静，等待连续 3 次问题监听后自动退出"

    Invoke-TestStep `
        -Title "6. 常驻演示测试" `
        -Command "python main_cli.py --wake-loop --wake-rounds 0" `
        -Prompt "请至少完成 2 轮对话，然后按 Ctrl+C 退出"
}
finally {
    Add-Content -LiteralPath $script:ResolvedLogPath -Value "===== Host voice test finished: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') =====" -Encoding UTF8
    Write-Host ""
    Write-Host "测试日志已保存到: $script:ResolvedLogPath"
}
