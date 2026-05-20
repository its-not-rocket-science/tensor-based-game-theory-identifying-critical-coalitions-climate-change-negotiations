import re
text = open('main.tex', encoding='utf-8').read()
text = re.sub(r'\\author\[1\].*?(?=\\date\{)', '', text, flags=re.DOTALL)
text = re.sub(r'\\affil\[1\]\{[^}]+\}\n', '', text)
text = re.sub(r'\\date\{[^}]*\}', r'\\date{}', text)
text = re.sub(r'\\href\{https://github[.]com/[^}]+\}\{(?:[^{}]|\{[^{}]*\})*\}',
              '[details withheld for double-anonymous review]', text)
text = re.sub(r'\\url\{https://github[.]com/[^}]+\}',
              '[details withheld for double-anonymous review]', text)
text = re.sub(
    r'(\\section\*\{Acknowledgments\}\n).*?(?=\n\\clearpage)',
    r'\1Blinded for review.\n',
    text, flags=re.DOTALL
)
open('main_anon.tex', 'w', encoding='utf-8').write(text)
print("main_anon.tex written.")
