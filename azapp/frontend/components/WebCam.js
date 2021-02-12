app.component('webcam', {
    template:
		/*html*/
		`
		<video ref="camera" :width="640" :height="480" autoplay></video>
		<canvas style="display:none;" ref="canvas" :width="640" :height="480"></canvas>
		`,
				
		mounted() {
			this.createCameraElement();
		},

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