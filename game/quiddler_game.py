import math
import pygtrie as trie

card_points = {'a':2,'b':8,'c':8,'d':5,'e':2,'f':6,'g':6,'h':7,'i':2,'j':13,'k':8,
                'l':3,'m':5,'n':5,'o':2,'p':6,'q':15,'r':5,'s':3,'t':3,'u':4,'v':11,
                'w':10,'x':12,'y':4,'z':14,'qu':9,'in':7,'er':7,'cl':10,'th':9}
cards = set(card_points.keys())
double_cards = set(list(card_points.keys())[-5:])
single_cards = set(list(card_points.keys())[:-5])
double_card_first = {d[0]:d[1] for d in double_cards}

# Build data structure for all possible words
with open("sowpods.txt", "r") as text_file:
    lines = text_file.readlines()
words = [w.strip() for w in lines[6:]]
print(len(lines))

def make_perm(s, clist, pos, collector):
    # move forward in the list through s until hitting a branching point when we call ourselves with copy of clist
    while pos < len(s):
        if pos < len(s)-1:
            # check for a double
            dc = s[pos]+s[pos+1]
            if dc in double_cards:
                make_perm(s, clist.copy()+[dc], pos+2, collector)
        clist+=[s[pos]]
        pos+=1
    # words have to be made from 2 to 10 cards
    if len(clist) >= 2 and len(clist) <= 10:
        collector['/'.join(clist)] = sum([card_points[c] for c in clist])

perms = {}
for s in words:
    make_perm(s,[],0,perms)

t = trie.StringTrie(separator='/')
t.update(perms)

def find_word(wt, p, hand_left, collector):
    for idx,c in enumerate(hand_left):
        h=hand_left.copy()
        tp = p+[h.pop(idx)]
        s = '/'.join(tp)
        if wt.has_subtrie(s):
            score = wt.get(s)
            if score:
                collector[s] = (score, len(tp), tp)
            find_word(wt,tp,h,collector)

# Find all possible words for a hand
hand = ['x','k','r','s','u','m','th','e','s','a']
possible_words = {}
find_word(t,[],hand,possible_words)

len(possible_words), sorted(possible_words.values(),reverse=True)

# Find best scoring combination
# first find all the valid combinations from the possible list
# then for each combo subtract points for unused cards (remember this subtraction amount)
pwt = trie.StringTrie(separator='/')
pwt.update({x[0]: x[1][0] for x in possible_words.items()})

def scored_play(combo, hand_left):
    ts, tc = 0, 0
    for e in combo:
        ts += e[1][0]
        tc += e[1][1]
    if len(hand_left) == 0:
        return (ts, True, combo)
    else:
        return (ts-sum([card_points[c] for c in hand_left]), False, combo)   

def find_word_combos(wt, combo, hand_left, collector):
    pw = {}
    find_word(wt,[],hand_left,pw)
    for w in pw.items():
        j = hand_left.copy()
        for c in w[1][2]:
            j.remove(c)
        cc = combo.copy()
        cc.append(w)
        find_word_combos(wt,cc,j,collector)
    collector.append(scored_play(combo,hand_left))


combos = []
for w in possible_words.items():
    j = hand.copy()
    for c in w[1][2]:
        j.remove(c)
    find_word_combos(pwt,[w],j,combos)

sorted(combos, reverse=True)[0]

def get_play(hand):
    possible_words = {}
    find_word(t,[],hand,possible_words)
    combos = []
    for w in possible_words.items():
        j = hand.copy()
        for c in w[1][2]:
            j.remove(c)
        find_word_combos(pwt,[w],j,combos)
    return sorted(combos, reverse=True)[0]

hand = ['e','e','a','y','qu','cl','n','u','w','s']

no_pick_up = get_play(hand)
options = [(no_pick_up, pick_up, pick_up)]
pick_up = 'h'
for idx,h in enumerate(hand):
    hc = hand.copy()
    hc[idx] = pick_up
    options.append((get_play(hc), pick_up, hand[idx]))
play = sorted(options, reverse=True)[0]

play

print(f'Score: {play[0][0]}')
print(f'Complete: {play[0][1]}')
print(f'Words: {[s[0] for s in play[0][2]]}')
print(f'Pick up: {play[1]}')
print(f'Drop: {play[2]}')


