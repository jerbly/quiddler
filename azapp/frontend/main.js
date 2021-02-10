//const apiBaseUrl = 'http://localhost:7071/api/'
const apiBaseUrl = 'https://quiddler.azurewebsites.net/api/'

const app = Vue.createApp({
    data() {
        return {
            play: {},
            hand: '',
            deck: '',
            handImage: '',
            deckImage: '',
            nCards: 4,
            waiting: false
        }
    },
    methods: {
        onHandChange(e) {
            this.fileChange(e, 'handImage');
        },
        onDeckChange(e) {
            this.fileChange(e, 'deckImage');
        },
        fileChange(e, im) {
            const file = e.target.files[0];
            this.createBase64Image(file, im);
        },
        createBase64Image(fileObject, im) {
            const reader = new FileReader();
            reader.onload = (e) => {
                this[im] = e.target.result;
            };
            reader.readAsDataURL(fileObject);
        },
        uploadImages() {
            this.waiting = true;
            axios.post(apiBaseUrl+'cards', {
                    n_cards: this.nCards, 
                    images: true,
                    hand: this.handImage,
                    deck: this.deckImage
                })
                .then((response) => {
                    this.waiting = false;
                    this.hand = response.data[0];
                    this.deck = response.data[1];
                    this.submitCards()
                })
                .catch((err) => {
                    return new Error(err.message);
                })
        },
        submitCards() {
            this.waiting = true;
            axios.post(apiBaseUrl+'game', {
                hand: this.hand,
                deck: this.deck
            })
            .then((response) => {
                this.play = response.data;
                this.waiting = false;
            })
        }
    }
})