# Source - https://stackoverflow.com/a/48280566
# Posted by Rakesh, modified by community. See post 'Timeline' for change history
# Retrieved 2026-05-01, License - CC BY-SA 4.0

from autocorrect import Speller

spell = Speller(lang='en')

print(spell('caaaar'))
print(spell('mussage'))
print(spell('survice'))
print(spell('wht sevice do u offr'))
