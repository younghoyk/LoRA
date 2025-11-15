import { app } from "/scripts/app.js";
import {CUSTOM_INT, recursiveLinkUpstream, transformFunc, swapInputs, renameNodeInputs, removeNodeInputs, getDrawColor} from "./utils.js"

function addMultiAreaConditioningCanvas(node) {

	const widget = {
		type: "customCanvas",
		name: "MultiAreaConditioning-Canvas",
		computeSize: function() {
			// Calculate and return fixed height to prevent overlap
			const totalNodeHeight = node.size[1];
			const titleHeight = LiteGraph.NODE_TITLE_HEIGHT;
			const inputsHeight = LiteGraph.NODE_WIDGET_HEIGHT * Math.max(node.inputs.length, node.outputs.length);

			// Count other widgets (7 widgets: resolutionX, resolutionY, index, x, y, width, height, strength)
			const otherWidgetsCount = 8;
			const otherWidgetsHeight = otherWidgetsCount * LiteGraph.NODE_WIDGET_HEIGHT;

			// Calculate available space for canvas
			let canvasHeight = totalNodeHeight - titleHeight - inputsHeight - otherWidgetsHeight - 40;

			// Minimum height
			if (canvasHeight < 150) canvasHeight = 150;

			// Store for use in draw
			this._computedHeight = canvasHeight;

			return [0, canvasHeight];
		},
		draw: function (ctx, node, widgetWidth, widgetY) {

			// Use the computed height from computeSize
			let widgetHeight = this._computedHeight || 200;

			const margin = 10
			const border = 2
            const values = node.properties["values"]
			const width = Math.round(node.properties["width"])
			const height = Math.round(node.properties["height"])

			const scale = Math.min((widgetWidth-margin*2)/width, (widgetHeight-margin*2)/height)

			const index = Math.round(node.widgets[3].value) // index widget is at position 3

            let backgroudWidth = width * scale
            let backgroundHeight = height * scale

			let xOffset = margin
			if (backgroudWidth < widgetWidth) {
				xOffset += (widgetWidth-backgroudWidth)/2 - margin
			}
			let yOffset = margin
			if (backgroundHeight < widgetHeight) {
				yOffset += (widgetHeight-backgroundHeight)/2 - margin
			}

			let widgetX = xOffset
			let drawY = widgetY + yOffset

			ctx.fillStyle = "#000000"
			ctx.fillRect(widgetX-border, drawY-border, backgroudWidth+border*2, backgroundHeight+border*2)

			ctx.fillStyle = globalThis.LiteGraph.NODE_DEFAULT_BGCOLOR
			ctx.fillRect(widgetX, drawY, backgroudWidth, backgroundHeight);

			function getDrawArea(v) {
				let x = v[0]*backgroudWidth/width
				let y = v[1]*backgroundHeight/height
				let w = v[2]*backgroudWidth/width
				let h = v[3]*backgroundHeight/height

				if (x > backgroudWidth) { x = backgroudWidth}
				if (y > backgroundHeight) { y = backgroundHeight}

				if (x+w > backgroudWidth) {
					w = Math.max(0, backgroudWidth-x)
				}
				
				if (y+h > backgroundHeight) {
					h = Math.max(0, backgroundHeight-y)
				}

				return [x, y, w, h]
			}
            
			// Draw all the conditioning zones
			for (const [k, v] of values.entries()) {

				if (k == index) {continue}

				const [x, y, w, h] = getDrawArea(v)

				ctx.fillStyle = getDrawColor(k/values.length, "80") //colors[k] + "B0"
				ctx.fillRect(widgetX+x, drawY+y, w, h)

			}

			ctx.beginPath();
			ctx.lineWidth = 1;

			for (let x = 0; x <= width/64; x += 1) {
				ctx.moveTo(widgetX+x*64*scale, drawY);
				ctx.lineTo(widgetX+x*64*scale, drawY+backgroundHeight);
			}

			for (let y = 0; y <= height/64; y += 1) {
				ctx.moveTo(widgetX, drawY+y*64*scale);
				ctx.lineTo(widgetX+backgroudWidth, drawY+y*64*scale);
			}

			ctx.strokeStyle = "#00000050";
			ctx.stroke();
			ctx.closePath();

			// Draw currently selected zone
			// console.log(index) // Removed debug output
			let [x, y, w, h] = getDrawArea(values[index])

			w = Math.max(32*scale, w)
			h = Math.max(32*scale, h)

			//ctx.fillStyle = "#"+(Number(`0x1${colors[index].substring(1)}`) ^ 0xFFFFFF).toString(16).substring(1).toUpperCase()
			ctx.fillStyle = "#ffffff"
			ctx.fillRect(widgetX+x, drawY+y, w, h)

			const selectedColor = getDrawColor(index/values.length, "FF")
			ctx.fillStyle = selectedColor
			ctx.fillRect(widgetX+x+border, drawY+y+border, w-border*2, h-border*2)

			// Display input indicator on node context
			ctx.beginPath();

			ctx.arc(LiteGraph.NODE_SLOT_HEIGHT*0.5, LiteGraph.NODE_SLOT_HEIGHT*(index + 0.5)+4, 4, 0, Math.PI * 2);
			ctx.fill();

			ctx.lineWidth = 1;
			ctx.strokeStyle = "white";
			ctx.stroke();

			// Connected node highlighting is drawn on the main context
			if (node.selected) {
				const connectedNodes = recursiveLinkUpstream(node, node.inputs[index].type, 0, index)

				if (connectedNodes.length !== 0) {
					for (let [node_ID, depth] of connectedNodes) {
						let connectedNode = node.graph._nodes_by_id[node_ID]
						if (connectedNode.type != node.type) {
							const [x, y] = connectedNode.pos
							const [w, h] = connectedNode.size
							const offset = 5
							const titleHeight = LiteGraph.NODE_TITLE_HEIGHT * (connectedNode.type === "Reroute"  ? 0 : 1)

							ctx.strokeStyle = selectedColor
							ctx.lineWidth = 5;
							ctx.strokeRect(x-offset-node.pos[0], y-offset-node.pos[1]-titleHeight, w+offset*2, h+offset*2+titleHeight)
						}
					}
				}
			}
			ctx.lineWidth = 1;
			ctx.closePath();

		},
	};

	node.addCustomWidget(widget);

	// Remove onResize to prevent auto-growing
	// Size will be managed by user manual resize only

	return { minWidth: 200, minHeight: 200, widget }
}

app.registerExtension({
	name: "Comfy.Davemane42.MultiAreaConditioning",
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
		if (nodeData.name === "MultiAreaConditioning") {
			const onNodeCreated = nodeType.prototype.onNodeCreated;
			nodeType.prototype.onNodeCreated = function () {
				const r = onNodeCreated ? onNodeCreated.apply(this, arguments) : undefined;

				this.setProperty("width", 512)
				this.setProperty("height", 512)
				this.setProperty("values", [[0, 0, 0, 0, 1.0], [0, 0, 0, 0, 1.0]])

				this.selected = false
				// Store index widget position for reference (but we'll use direct access widgets[3])
				// this.index = 3

                this.serialize_widgets = true;

				CUSTOM_INT(this, "resolutionX", 512, function (v, _, node) {const s = this.options.step / 10; this.value = Math.round(v / s) * s; node.properties["width"] = this.value})
				CUSTOM_INT(this, "resolutionY", 512, function (v, _, node) {const s = this.options.step / 10; this.value = Math.round(v / s) * s; node.properties["height"] = this.value})

				addMultiAreaConditioningCanvas(this)

				// Canvas height will be calculated dynamically in draw function

				CUSTOM_INT(
					this,
					"index",
					0,
					function (v, _, node) {
						console.log("Index changed to:", v);
						console.log("Node widgets:", node.widgets.map((w, i) => `${i}: ${w.name}`));
						console.log("Values array:", node.properties["values"]);

						let values = node.properties["values"]

						if (values && values[v]) {
							console.log(`Loading values[${v}]:`, values[v]);
							// Widget positions: 0=resolutionX, 1=resolutionY, 2=canvas, 3=index
							// 4=x, 5=y, 6=width, 7=height, 8=strength
							node.widgets[4].value = values[v][0]  // x
							node.widgets[5].value = values[v][1]  // y
							node.widgets[6].value = values[v][2]  // width
							node.widgets[7].value = values[v][3]  // height
							if (!values[v][4]) {values[v][4] = 1.0}
							node.widgets[8].value = values[v][4]  // strength

							// Force UI update
							node.setDirtyCanvas(true);
						} else {
							console.warn("Invalid index or values not found", v, values);
						}
					},
					{ step: 10, max: 1 }

				)

				CUSTOM_INT(this, "x", 0, function (v, _, node) {transformFunc(this, v, node, 0)})
				CUSTOM_INT(this, "y", 0, function (v, _, node) {transformFunc(this, v, node, 1)})
				CUSTOM_INT(this, "width", 0, function (v, _, node) {transformFunc(this, v, node, 2)})
				CUSTOM_INT(this, "height", 0, function (v, _, node) {transformFunc(this, v, node, 3)})
				CUSTOM_INT(this, "strength", 1, function (v, _, node) {transformFunc(this, v, node, 4)}, {"min": 0.0, "max": 10.0, "step": 0.1, "precision": 2})

				this.getExtraMenuOptions = function(_, options) {
					options.unshift(
						{
							content: `insert input above ${this.widgets[3].value} /\\`,
							callback: () => {
								this.addInput("conditioning", "CONDITIONING")
								
								const inputLenth = this.inputs.length-1
								const index = this.widgets[3].value

								for (let i = inputLenth; i > index; i--) {
									swapInputs(this, i, i-1)
								}
								renameNodeInputs(this, "conditioning")

								this.properties["values"].splice(index, 0, [0, 0, 0, 0, 1])
								this.widgets[3].options.max = inputLenth

								this.setDirtyCanvas(true);

							},
						},
						{
							content: `insert input below ${this.widgets[3].value} \\/`,
							callback: () => {
								this.addInput("conditioning", "CONDITIONING")
								
								const inputLenth = this.inputs.length-1
								const index = this.widgets[3].value

								for (let i = inputLenth; i > index+1; i--) {
									swapInputs(this, i, i-1)
								}
								renameNodeInputs(this, "conditioning")

								this.properties["values"].splice(index+1, 0, [0, 0, 0, 0, 1])
								this.widgets[3].options.max = inputLenth

								this.setDirtyCanvas(true);
							},
						},
						{
							content: `swap with input above ${this.widgets[3].value} /\\`,
							callback: () => {
								const index = this.widgets[3].value
								if (index !== 0) {
									swapInputs(this, index, index-1)

									renameNodeInputs(this, "conditioning")

									this.properties["values"].splice(index-1,0,this.properties["values"].splice(index,1)[0]);
									this.widgets[3].value = index-1

									this.setDirtyCanvas(true);
								}
							},
						},
						{
							content: `swap with input below ${this.widgets[3].value} \\/`,
							callback: () => {
								const index = this.widgets[3].value
								if (index !== this.inputs.length-1) {
									swapInputs(this, index, index+1)

									renameNodeInputs(this, "conditioning")
									
									this.properties["values"].splice(index+1,0,this.properties["values"].splice(index,1)[0]);
									this.widgets[3].value = index+1

									this.setDirtyCanvas(true);
								}
							},
						},
						{
							content: `remove currently selected input ${this.widgets[3].value}`,
							callback: () => {
								const index = this.widgets[3].value
								removeNodeInputs(this, [index])
								renameNodeInputs(this, "conditioning")
								// Update index widget max after removal
								this.widgets[3].options.max = this.properties["values"].length-1
								if (this.widgets[3].value > this.widgets[3].options.max) {
									this.widgets[3].value = this.widgets[3].options.max
								}
							},
						},
						{
							content: "remove all unconnected inputs",
							callback: () => {
								let indexesToRemove = []

								for (let i = 0; i < this.inputs.length; i++) {
									if (!this.inputs[i].link) {
										indexesToRemove.push(i)
									}
								}

								if (indexesToRemove.length) {
									removeNodeInputs(this, indexesToRemove, "conditioning")
								}
								renameNodeInputs(this, "conditioning")
							},
						},
					);
				}

				this.onRemoved = function () {
					// Cleanup function if needed
				};
			
				this.onSelected = function () {
					this.selected = true
				}
				this.onDeselected = function () {
					this.selected = false
				}

				return r;
			};
		}
	},
	loadedGraphNode(node, _) {
		if (node.type === "MultiAreaConditioning") {
			// index widget is at position 3
			if (node.widgets[3]) {
				node.widgets[3].options["max"] = node.properties["values"].length-1
				console.log("Set index max to:", node.properties["values"].length-1);
			}
		}
	},
	
});