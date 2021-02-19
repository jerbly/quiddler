app.component('webcam', {
    template:
		/*html*/
		`
		<button v-show="!cameraOpen" class="btn btn-success" type="button" @click="createCameraElement()">Open camera</button>
		<video v-show="cameraOpen" ref="camera" :width="320" :height="240" autoplay></video>
		<canvas style="display:none;" ref="canvas" :width="640" :height="480"></canvas>
		`,
				
		data() {
			return {
				cameraOpen: false
			}
		},
		emits: ['camera-opened'],
		methods: {
			createCameraElement() {
				const constraints = (window.constraints = {
					audio: false,
					video: true
				});

				navigator.mediaDevices
					.getUserMedia(constraints)
					.then(stream => {
						this.$refs.camera.srcObject = stream;
						this.cameraOpen = true;
						this.$emit('camera-opened');
					})
					.catch(error => {
							alert(error);
					});
			},
		
			takePhoto(im) {
				const context = this.$refs.canvas.getContext('2d');
				context.drawImage(this.$refs.camera, 0, 0, 640, 480);
				const dataURL = this.$refs.canvas.toDataURL();
				this.$parent[im] = dataURL;
			}
		}
})