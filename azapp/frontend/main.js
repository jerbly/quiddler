//const apiBaseUrl = 'http://localhost:7071/api/'
const apiBaseUrl = 'https://quiddler.azurewebsites.net/api/'
const emptyImage = 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7'

const app = Vue.createApp({
    data() {
        return {
            play: {},
            hand: '',
            deck: '',
            handImage: '',
            deckImage: '',
            nCards: 4,
            waiting: false,
            state: ''
        }
    },
    mounted() {
        this.$refs.handImage.src = emptyImage;
        this.$refs.deckImage.src = emptyImage;
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
            this.state = 'Reading images...'
            this.waiting = true;
            axios.post(apiBaseUrl+'cards', {
                    n_cards: this.nCards, 
                    images: true,
                    hand: this.handImage,
                    deck: this.deckImage
                })
                .then((response) => {
                    this.waiting = false;
                    //console.log(response.data)
                    this.hand = response.data[0];
                    this.deck = response.data[1];
                    if (this.hand && this.deck) {
                        this.handImage = response.data[2];
                        this.deckImage = response.data[3];
                        this.submitCards();
                    }
                })
                .catch((err) => {
                    this.waiting = false;
                    return new Error(err.message);
                })
        },
        submitCards() {
            this.state = 'Finding the best play...'
            this.waiting = true;
            axios.post(apiBaseUrl+'game', {
                hand: this.hand,
                deck: this.deck
            })
            .then((response) => {
                this.waiting = false;
                if (response.data) {
                    this.play = response.data;
                }
            })
        }
    },
    computed: {
        wordLinks() {
            if (this.play.words) {
                return this.play.words;
            }
            return '';
        }
    }
})