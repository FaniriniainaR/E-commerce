# shop/templatetags/dict_utils.py
from django import template

register = template.Library()

@register.filter(is_safe=True)
def dict_length(value):
    return len(value)