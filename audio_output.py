"""
固定提示音 wav 文件播放（区分平台：Windows winsound / Linux ALSA aplay）。

并提供 Linux / SC171 板端的声卡索引查找与 amixer 通路配置，
供本模块和 streaming_tts.py 的连续 PCM 播放复用。
"""

from __future__ import annotations

import logging
import platform
import shutil
import subprocess
from pathlib import Path


logger = logging.getLogger(__name__)


def get_sound_card_index(card_name: str = "lahainayupikiot") -> str | None:
    """在 Linux 下查找目标声卡索引。"""
    try:
        ret = subprocess.run(
            ["cat", "/proc/asound/cards"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for line in ret.stdout.splitlines():
            if card_name in line:
                return line.split()[0]
        return None
    except Exception as exc:
        logger.warning("获取声卡索引失败: %s", exc)
        return None


def set_sound_mixer_command(index: str) -> None:
    """按板端常见配置设置音频通路。"""
    commands = [
        ["amixer", "-c", str(index), "cset", "numid=243,iface=MIXER,name='RX HPH Mode'", "CLS_AB"],
        ["amixer", "-c", str(index), "cset", "numid=90,iface=MIXER,name='RX_MACRO RX0 MUX'", "AIF1_PB"],
        ["amixer", "-c", str(index), "cset", "numid=91,iface=MIXER,name='RX_MACRO RX1 MUX'", "AIF1_PB"],
        ["amixer", "-c", str(index), "cset", "numid=6639,iface=MIXER,name='RX_CDC_DMA_RX_0 Channels'", "Two"],
        ["amixer", "-c", str(index), "cset", "numid=112,iface=MIXER,name='RX INT0_1 MIX1 INP0'", "RX0"],
        ["amixer", "-c", str(index), "cset", "numid=115,iface=MIXER,name='RX INT1_1 MIX1 INP0'", "RX1"],
        ["amixer", "-c", str(index), "cset", "numid=107,iface=MIXER,name='RX INT0 DEM MUX'", "CLSH_DSM_OUT"],
        ["amixer", "-c", str(index), "cset", "numid=108,iface=MIXER,name='RX INT1 DEM MUX'", "CLSH_DSM_OUT"],
        ["amixer", "-c", str(index), "cset", "numid=137,iface=MIXER,name='RX_COMP1 Switch'", "1"],
        ["amixer", "-c", str(index), "cset", "numid=138,iface=MIXER,name='RX_COMP2 Switch'", "1"],
        ["amixer", "-c", str(index), "cset", "numid=244,iface=MIXER,name='HPHL_COMP Switch'", "1"],
        ["amixer", "-c", str(index), "cset", "numid=245,iface=MIXER,name='HPHR_COMP Switch'", "1"],
        ["amixer", "-c", str(index), "cset", "numid=269,iface=MIXER,name='HPHL_RDAC Switch'", "1"],
        ["amixer", "-c", str(index), "cset", "numid=270,iface=MIXER,name='HPHR_RDAC Switch'", "1"],
        ["amixer", "-c", str(index), "cset", "numid=520,iface=MIXER,name='RX_CDC_DMA_RX_0 Audio Mixer MultiMedia1'", "1"],
    ]

    for cmd in commands:
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        except subprocess.CalledProcessError as exc:
            logger.debug("配置混音器失败: %s", exc.stderr)


def play_wav_file(card_index: str, wav_path: str) -> bool:
    """使用 aplay 播放 wav。"""
    cmd = [
        "aplay",
        "-t",
        "wav",
        "-r",
        "48000",
        "-f",
        "S16_BE",
        "-c",
        "2",
        "-D",
        f"hw:{card_index},0",
        wav_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return True
    except Exception as exc:
        logger.error("播放失败: %s", exc)
        return False


class AudioOutput:
    """统一音频播放入口。"""

    def __init__(self) -> None:
        self.is_linux = platform.system().lower() == "linux"

    def play_wav(self, wav_path: str | Path) -> bool:
        audio_path = Path(wav_path)
        if not audio_path.exists():
            logger.error("待播放音频不存在: %s", audio_path)
            return False

        if self.is_linux:
            return self._play_on_linux(audio_path)
        return self._play_on_non_linux(audio_path)

    def _play_on_linux(self, audio_path: Path) -> bool:
        card_index = get_sound_card_index()
        if card_index:
            set_sound_mixer_command(card_index)
            if play_wav_file(card_index, str(audio_path)):
                logger.info("已通过 ALSA 播放音频: %s", audio_path)
                return True

        if shutil.which("aplay"):
            try:
                subprocess.run(["aplay", str(audio_path)], check=True)
                logger.info("已通过系统 aplay 播放音频: %s", audio_path)
                return True
            except Exception as exc:
                logger.error("系统 aplay 播放失败: %s", exc)

        logger.error("未找到可用的 Linux 音频播放方式")
        return False

    def _play_on_non_linux(self, audio_path: Path) -> bool:
        if platform.system().lower() == "windows":
            try:
                import winsound

                winsound.PlaySound(str(audio_path), winsound.SND_FILENAME)
                logger.info("已通过 Windows winsound 播放音频: %s", audio_path)
                return True
            except Exception as exc:
                logger.error("Windows 音频播放失败: %s", exc)
                return False

        logger.warning("当前系统暂未实现自动播放。音频文件保留在: %s", audio_path)
        return False
