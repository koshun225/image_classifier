"""
カスタムテンプレートフィルター
"""
import os
from django import template

register = template.Library()


@register.filter
def basename(value):
    """ファイルパスからファイル名を取得"""
    if value:
        return os.path.basename(str(value))
    return value


@register.filter
def filename_short(value, max_length=30):
    """ファイル名を短縮表示"""
    name = basename(value)
    if len(name) > max_length:
        return name[:max_length-3] + '...'
    return name

