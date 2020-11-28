import pygtrie as trie

card_points = {'a':2,'b':8,'c':8,'d':5,'e':2,'f':6,'g':6,'h':7,'i':2,'j':13,'k':8,
                'l':3,'m':5,'n':5,'o':2,'p':6,'q':15,'r':5,'s':3,'t':3,'u':4,'v':11,
                'w':10,'x':12,'y':4,'z':14,'qu':9,'in':7,'er':7,'cl':10,'th':9}
cards = set(card_points.keys())
double_cards = set(list(card_points.keys())[-5:])
single_cards = set(list(card_points.keys())[:-5])
double_card_first = {d[0]:d[1] for d in double_cards}

def make_perm(word, card_list, pos, collector):
    '''
    Given a word, break it into all the ways to make that word using the cards and get the
    total score for each permutation.

    For example the word: "inquiring" can be constructed from the cards in 8 ways: 
        {'in/qu/i/r/in/g': 36,
        'in/qu/i/r/i/n/g': 36,
        'in/q/u/i/r/in/g': 46,
        'in/q/u/i/r/i/n/g': 46,
        'i/n/qu/i/r/in/g': 36,
        'i/n/qu/i/r/i/n/g': 36,
        'i/n/q/u/i/r/in/g': 46,
        'i/n/q/u/i/r/i/n/g': 46}

    This is a recursive function.
        card_list holds the cards as they're identified for this word
        pos is the character position through the word
        collector is a dict where the results are stored
    '''
    while pos < len(word):
        if pos < len(word)-1:
            # check for a double letter card
            dc = word[pos]+word[pos+1]
            if dc in double_cards:
                # found a double letter card so fork here
                make_perm(word, card_list.copy()+[dc], pos+2, collector)
        card_list+=[word[pos]]
        pos+=1
    # words have to be made from 2 to 10 cards so only collect these
    if len(card_list) >= 2 and len(card_list) <= 10:
        collector['/'.join(card_list)] = sum([card_points[c] for c in card_list])

# Construct the Prefix Tree for all possible word/card permutations
with open("sowpods.txt", "r") as text_file:
    lines = text_file.readlines()
perms = trie.StringTrie(separator='/')
for w in lines[6:]:
    make_perm(w.strip(),[],0,perms)
print(f'Using {len(perms)} possible word permutations\n')

def find_word(wt, p, hand_left, collector):
    for idx, _ in enumerate(hand_left):
        h=hand_left.copy()
        tp = p+[h.pop(idx)]
        s = '/'.join(tp)
        if wt.has_subtrie(s):
            score = wt.get(s)
            if score:
                collector[s] = (score, len(tp), tp)
            find_word(wt,tp,h,collector)

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

def get_play(hand):
    possible_words = {}
    find_word(perms,[],hand,possible_words)
    pwt = trie.StringTrie(separator='/')
    pwt.update({x[0]: x[1][0] for x in possible_words.items()})
    combos = []
    for w in possible_words.items():
        j = hand.copy()
        for c in w[1][2]:
            j.remove(c)
        find_word_combos(pwt,[w],j,combos)
    return sorted(combos, reverse=True)[0] if len(combos) > 0 else None

def get_best_play(hand, pick_up):
    print(f'Hand: {"/".join(hand)}')
    print(f'Deck: {pick_up}')
    no_pick_up = get_play(hand)
    options = []
    # Tuple is: (Hand, Card picked up, Card dropped)
    # First option is to not pick up hence the card picked up and dropped are the same
    if no_pick_up: 
        options.append((no_pick_up, pick_up, pick_up))
    for idx, _ in enumerate(hand):
        hc = hand.copy()
        hc[idx] = pick_up
        p = get_play(hc)
        if p:
            options.append((p, pick_up, hand[idx]))
    if options:
        play = sorted(options, reverse=True)[0]
        if play[0]:
            print(f'Score: {play[0][0]}')
            print(f'Complete: {play[0][1]}')
            print(f'Words: {[s[0] for s in play[0][2]]}')
        print(f'Pick up: {play[1]}')
        print(f'Drop: {play[2]}\n')


get_best_play(['qu','e','b','n','z'],'s')
get_best_play(['i','c','e','v','i','s','i','o','n'],'i')
