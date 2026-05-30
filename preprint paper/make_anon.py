import re
text = open('main.tex', encoding='utf-8').read()
text = re.sub(r'\\author\[1\].*?(?=\\date\{)', r'\\author{}\n', text, flags=re.DOTALL)
text = re.sub(r'\\affil\[1\]\{[^}]+\}\n', '', text)
text = re.sub(r'\\date\{[^}]*\}', r'\\date{}', text)
text = re.sub(r'\\href\{https://github[.]com/[^}]+\}\{(?:[^{}]|\{[^{}]*\})*\}',
              '[details withheld for double-anonymous review]', text)
text = re.sub(r'\\url\{https://github[.]com/[^}]+\}',
              '[details withheld for double-anonymous review]', text)
# Remove Data Availability section entirely (ERE double-blind requirement)
text = re.sub(
    r'\\section\*\{Data Availability\}.*?(?=\\section\*\{)',
    '',
    text, flags=re.DOTALL
)
# Remove Acknowledgments section entirely (ERE double-blind requirement)
text = re.sub(
    r'\\section\*\{Acknowledgments\}.*?(?=\\appendix)',
    '',
    text, flags=re.DOTALL
)
# Remove Appendix (ERE requirement: provide as separate supplementary file)
text = re.sub(
    r'\\appendix.*?(?=\\clearpage)',
    '',
    text, flags=re.DOTALL
)
open('main_anon.tex', 'w', encoding='utf-8').write(text)
print("main_anon.tex written.")
