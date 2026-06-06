import { app } from '../../../scripts/app.js'

// Helper function to create a unique ID
export function makeUUID() {
  let dt = new Date().getTime()
  const uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = ((dt + Math.random() * 16) % 16) | 0
    dt = Math.floor(dt / 16)
    return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16)
  })
  return uuid
}

function chainCallback(object, property, callback) {
  if (object == undefined) {
    console.error("Tried to add callback to non-existant object")
    return;
  }
  if (property in object) {
    const callback_orig = object[property]
    object[property] = function () {
      const r = callback_orig.apply(this, arguments);
      callback.apply(this, arguments);
      return r
    };
  } else {
    object[property] = callback;
  }
}

// Shared initialization function for both SimpleFastPreview and AdvancedFastPreview
function initializePreviewNode(nodeType, nodeData) {
  chainCallback(nodeType.prototype, "onNodeCreated", function () {

    var element = document.createElement("div");
    this.uuid = makeUUID()
    element.id = `fast-preview-${this.uuid}`

    this.previewWidget = this.addDOMWidget(nodeData.name, "FastPreviewWidget", element, {
      serialize: false,
      hideOnZoom: false,
    });

    this.previewer = new Previewer(this);

    this.setSize([512 + 20, 512 + 40]); // Initial size
    this.resizable = true; // Allow resizing

    // This is a hack to get the node's content area and apply flex layout.
    // This might break with future ComfyUI updates.
    if (!this.flex_layout_applied) {
      try {
        // The element created by addDOMWidget.
        const widgetElement = this.previewWidget.element;
        // The container for our widget.
        const widgetContainer = widgetElement.parentElement;
        // The area containing all widgets.
        const widgetsArea = widgetContainer.parentElement;
        // The main content area of the node.
        const contentArea = widgetsArea.parentElement;

        contentArea.style.display = 'flex';
        contentArea.style.flexDirection = 'column';

        // Allow the preview widget to grow and fill the remaining space.
        widgetContainer.style.flexGrow = 1;
        widgetContainer.style.minHeight = '100px';

        this.flex_layout_applied = true;
      } catch (e) {
        console.error("TKSW_NODE: Failed to apply flex layout to preview node:", e);
      }
    }

    this.previewWidget.parentEl = document.createElement("div");
    this.previewWidget.parentEl.className = "fast-preview-container";
    this.previewWidget.parentEl.style.width = `100%`;
    this.previewWidget.parentEl.style.height = `100%`;
    this.previewWidget.parentEl.style.position = 'relative';
    element.appendChild(this.previewWidget.parentEl);

    const previewEl = document.createElement("div");
    previewEl.className = "fast-preview";
    previewEl.style.width = '100%';
    previewEl.style.height = '100%'; // Fill the container
    this.previewWidget.parentEl.appendChild(previewEl);
    this.previewer.previewEl = previewEl;

    this.previewer.createHistoryButtons();

    chainCallback(this, "onExecuted", function (message) {
      let bg_image = message["bg_image"];
      this.previewer.addImageToHistory(bg_image);
    });
  });
}

app.registerExtension({
  name: 'TestNode.FastPreview',

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name === 'SimpleFastPreview' || nodeData?.name === 'AdvancedFastPreview') {
      initializePreviewNode(nodeType, nodeData);
    }
  }
})

class Previewer {
  constructor(context) {
    this.node = context;
    this.history = [];
    this.historyIndex = -1;
    this.maxHistory = 20;
    this.previewEl = null;
    this.historyLabel = null;
  }

  createHistoryButtons = () => {
    const buttonContainer = document.createElement("div");
    buttonContainer.className = "history-buttons";
    buttonContainer.style.position = "absolute";
    buttonContainer.style.bottom = "10px";
    buttonContainer.style.left = "50%";
    buttonContainer.style.transform = "translateX(-50%)";
    buttonContainer.style.display = "flex";
    buttonContainer.style.gap = "10px";
    buttonContainer.style.zIndex = "10";

    const prevButton = document.createElement("button");
    prevButton.innerText = "<";
    prevButton.onclick = () => this.navigateHistory(-1);

    this.historyLabel = document.createElement("span");
    this.historyLabel.innerText = "-/-";
    this.historyLabel.style.lineHeight = "1.5";
    this.historyLabel.style.verticalAlign = "middle";


    const nextButton = document.createElement("button");
    nextButton.innerText = ">";
    nextButton.onclick = () => this.navigateHistory(1);

    buttonContainer.appendChild(prevButton);
    buttonContainer.appendChild(this.historyLabel);
    buttonContainer.appendChild(nextButton);

    this.node.previewWidget.parentEl.appendChild(buttonContainer);
  }

  updateHistoryLabel = () => {
    if (this.history.length > 0) {
      this.historyLabel.innerText = `${this.historyIndex + 1}/${this.history.length}`;
    } else {
      this.historyLabel.innerText = "-/-";
    }
  }

  navigateHistory = (direction) => {
    if (this.history.length === 0) return;
    const newIndex = this.historyIndex + direction;
    if (newIndex >= 0 && newIndex < this.history.length) {
      this.historyIndex = newIndex;
      this.refreshBackgroundImage();
    }
  }

  addImageToHistory = (base64Image) => {
    this.history.push(base64Image);
    if (this.history.length > this.maxHistory) {
      this.history.shift(); // Remove the oldest image
    }
    this.historyIndex = this.history.length - 1;
    this.refreshBackgroundImage();
  }

  refreshBackgroundImage = () => {
    if (this.historyIndex >= 0 && this.historyIndex < this.history.length) {
      const base64String = this.history[this.historyIndex];
      const imageUrl = `data:image/jpeg;base64,${base64String}`;
      
      this.previewEl.style.backgroundImage = `url(${imageUrl})`;
      this.previewEl.style.backgroundSize = 'contain';
      this.previewEl.style.backgroundRepeat = 'no-repeat';
      this.previewEl.style.backgroundPosition = 'center';
      
      this.updateHistoryLabel();
    }
  };
}