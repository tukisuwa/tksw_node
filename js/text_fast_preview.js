import { app } from "/scripts/app.js";

function chainCallback(object, property, callback) {
    if (object == undefined) {
        console.error("Tried to add callback to non-existant object");
        return;
    }
    if (property in object) {
        const callback_orig = object[property];
        object[property] = function () {
            const r = callback_orig?.apply(this, arguments);
            callback?.apply(this, arguments);
            return r;
        };
    } else {
        object[property] = callback;
    }
}

function extractScalar(value) {
    if (Array.isArray(value)) return value[0];
    return value;
}

function findWidget(node, name) {
    return node.widgets?.find((w) => w.name === name);
}

function hideWidget(widget) {
    if (!widget) return;
    if (widget._tksw_hidden) return;
    widget._tksw_hidden = true;
    widget.serialize = true;
    widget.type = "hidden";
    widget.computeSize = () => [0, 0];
    widget.draw = () => {};
    if (widget.element?.style) widget.element.style.display = "none";
    if (widget.inputEl?.style) widget.inputEl.style.display = "none";
}

function updateTextPreviewLayout(node) {
    const widget = node?.textPreviewWidget;
    const element = widget?.element;
    if (!widget || !element || !node?.size) return;

    const nodeWidth = Math.max(220, Number(node.size[0] ?? 520));
    const nodeHeight = Math.max(100, Number(node.size[1] ?? 320));
    const widgetTop = Number(widget.last_y ?? widget.y ?? 34);
    const width = Math.max(180, nodeWidth - 20);
    const height = Math.max(60, nodeHeight - widgetTop - 24);

    element.style.width = `${width}px`;
    element.style.height = `${height}px`;
    element.style.maxHeight = `${height}px`;

    const parent = node.textPreviewWidget?.parentEl;
    if (parent) {
        parent.style.height = `${height}px`;
        parent.style.maxHeight = `${height}px`;
    }
}

function scheduleTextPreviewLayout(node, attempts = 3) {
    requestAnimationFrame(() => {
        updateTextPreviewLayout(node);
        node?.setDirtyCanvas?.(true, true);
        if (attempts > 1) scheduleTextPreviewLayout(node, attempts - 1);
    });
}

function initializeTextPreviewNode(nodeType, nodeData) {
    chainCallback(nodeType.prototype, "onNodeCreated", function () {
        const storedTextWidget = findWidget(this, "text");
        hideWidget(storedTextWidget);
        const saveHistoryWidget = findWidget(this, "save_history");
        hideWidget(saveHistoryWidget);
        const historyWidget = findWidget(this, "history_json");
        hideWidget(historyWidget);

        const element = document.createElement("div");
        Object.assign(element.style, {
            width: "100%",
            height: "100%",
            minHeight: "0",
            overflow: "hidden",
            boxSizing: "border-box",
            position: "relative",
            display: "flex",
            flexDirection: "column",
        });

        this.textPreviewWidget = this.addDOMWidget(nodeData.name, "TextPreviewWidget", element, {
            serialize: false,
            hideOnZoom: false,
        });
        this.textPreviewWidget.computeSize = (width) => [Math.max(220, Number(width ?? this.size?.[0] ?? 520) - 20), 80];

        this.textPreviewer = new TextPreviewer(this, storedTextWidget, historyWidget, saveHistoryWidget);

        this.setSize([520, 320]);
        this.resizable = true;

        if (!this.flex_layout_applied) {
            try {
                const widgetElement = this.textPreviewWidget.element;
                const widgetContainer = widgetElement.parentElement;
                const widgetsArea = widgetContainer.parentElement;
                const contentArea = widgetsArea.parentElement;

                contentArea.style.display = "flex";
                contentArea.style.flexDirection = "column";
                contentArea.style.overflow = "hidden";
                contentArea.style.minHeight = "0";

                widgetsArea.style.flex = "1 1 auto";
                widgetsArea.style.minHeight = "0";
                widgetsArea.style.overflow = "hidden";

                widgetContainer.style.flex = "1 1 auto";
                widgetContainer.style.minHeight = "0";
                widgetContainer.style.height = "100%";
                widgetContainer.style.overflow = "hidden";

                this.flex_layout_applied = true;
            } catch (e) {
                console.error("TKSW_NODE: Failed to apply flex layout to text preview node:", e);
            }
        }

        this.textPreviewWidget.parentEl = document.createElement("div");
        this.textPreviewWidget.parentEl.className = "text-preview-container";
        this.textPreviewWidget.parentEl.style.width = "100%";
        this.textPreviewWidget.parentEl.style.height = "100%";
        this.textPreviewWidget.parentEl.style.minHeight = "0";
        this.textPreviewWidget.parentEl.style.position = "relative";
        this.textPreviewWidget.parentEl.style.display = "flex";
        this.textPreviewWidget.parentEl.style.flexDirection = "column";
        this.textPreviewWidget.parentEl.style.overflow = "hidden";
        this.textPreviewWidget.parentEl.style.boxSizing = "border-box";
        element.appendChild(this.textPreviewWidget.parentEl);

        const textarea = document.createElement("textarea");
        textarea.className = "text-preview";
        textarea.readOnly = true;
        textarea.spellcheck = false;
        textarea.wrap = "soft";
        textarea.style.width = "100%";
        textarea.style.height = "auto";
        textarea.style.flex = "1 1 auto";
        textarea.style.minHeight = "0";
        textarea.style.boxSizing = "border-box";
        textarea.style.padding = "8px 58px 38px 8px";
        textarea.style.fontFamily = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace";
        textarea.style.fontSize = "12px";
        textarea.style.resize = "none";
        textarea.style.overflow = "auto";
        textarea.style.overflowWrap = "anywhere";
        textarea.style.wordBreak = "break-word";
        this.textPreviewWidget.parentEl.appendChild(textarea);
        this.textPreviewer.textareaEl = textarea;

        this.textPreviewer.createHistoryButtons();
        this.textPreviewer.createSaveHistoryToggle();
        this.textPreviewer.hydrateFromStoredText();
        if (saveHistoryWidget) {
            const originalCallback = saveHistoryWidget.callback;
            saveHistoryWidget.callback = function () {
                const result = originalCallback?.apply(this, arguments);
                if (saveHistoryWidget.value === false) {
                    if (historyWidget) {
                        historyWidget.value = "";
                        historyWidget.callback?.("");
                    }
                } else {
                    this.textPreviewer?.syncHistoryWidget?.();
                }
                this.textPreviewer?.paintSaveHistoryButton?.();
                this.setDirtyCanvas?.(true, true);
                return result;
            }.bind(this);
        }
        scheduleTextPreviewLayout(this, 4);

        chainCallback(this, "onResize", function () {
            updateTextPreviewLayout(this);
            this.setDirtyCanvas?.(true, true);
        });

        chainCallback(this, "onExecuted", function (message) {
            const previewText = extractScalar(message?.preview_text);
            if (previewText != undefined) {
                this.textPreviewer.addTextToHistory(previewText);
            }
        });
    });
}

app.registerExtension({
    name: "tksw_node.TextFastPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name === "TextFastPreview") {
            initializeTextPreviewNode(nodeType, nodeData);
        }
    },
});

class TextPreviewer {
    constructor(context, storedTextWidget, historyWidget, saveHistoryWidget) {
        this.node = context;
        this.storedTextWidget = storedTextWidget ?? null;
        this.historyWidget = historyWidget ?? null;
        this.saveHistoryWidget = saveHistoryWidget ?? null;
        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 20;
        this.textareaEl = null;
        this.historyLabel = null;
        this.saveHistoryButton = null;
    }

    shouldSaveHistory = () => {
        return this.saveHistoryWidget?.value !== false;
    };

    hydrateFromStoredText = () => {
        if (this.hydrateFromHistoryWidget()) return;
        const initial = this.storedTextWidget?.value;
        if (initial != undefined && String(initial).length > 0) {
            this.addTextToHistory(initial);
            return;
        }
        this.refreshText();
    };

    hydrateFromHistoryWidget = () => {
        if (!this.shouldSaveHistory() || !this.historyWidget?.value) return false;
        try {
            const data = JSON.parse(this.historyWidget.value);
            const history = Array.isArray(data.history) ? data.history.map((item) => String(item ?? "")) : [];
            if (!history.length) return false;
            this.history = history.slice(-this.maxHistory);
            const fallbackIndex = this.history.length - 1;
            const parsedIndex = Number(data.index);
            this.historyIndex = Number.isFinite(parsedIndex)
                ? Math.max(0, Math.min(this.history.length - 1, Math.trunc(parsedIndex)))
                : fallbackIndex;
            this.refreshText({ syncHistory: false });
            return true;
        } catch (_) {
            return false;
        }
    };

    syncHistoryWidget = () => {
        if (!this.historyWidget) return;
        if (!this.shouldSaveHistory()) {
            if (this.historyWidget.value) {
                this.historyWidget.value = "";
                this.historyWidget.callback?.("");
                this.node?.setDirtyCanvas?.(true, false);
            }
            return;
        }
        const value = JSON.stringify({
            history: this.history,
            index: this.historyIndex,
        });
        if (this.historyWidget.value === value) return;
        this.historyWidget.value = value;
        this.historyWidget.callback?.(value);
        this.node?.setDirtyCanvas?.(true, false);
    };

    syncStoredTextWidget = (text) => {
        if (!this.storedTextWidget) return;
        const value = String(text ?? "");
        if (this.storedTextWidget.value === value) return;
        this.storedTextWidget.value = value;
        this.node?.setDirtyCanvas?.(true, false);
    };

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

        this.node.textPreviewWidget.parentEl.appendChild(buttonContainer);
    };

    createSaveHistoryToggle = () => {
        if (!this.saveHistoryWidget) return;
        const button = document.createElement("button");
        button.type = "button";
        button.title = "Save text history in workflow";
        button.onclick = () => {
            this.saveHistoryWidget.value = this.saveHistoryWidget.value === false;
            this.saveHistoryWidget.callback?.(this.saveHistoryWidget.value);
            this.paintSaveHistoryButton();
        };
        Object.assign(button.style, {
            position: "absolute",
            top: "8px",
            right: "8px",
            zIndex: "10",
            height: "20px",
            minWidth: "44px",
            padding: "1px 6px",
            border: "1px solid rgba(255,255,255,0.18)",
            borderRadius: "0",
            background: "rgba(0,0,0,0.35)",
            color: "var(--fg-color)",
            fontSize: "10px",
            lineHeight: "14px",
            boxSizing: "border-box",
        });
        this.saveHistoryButton = button;
        this.paintSaveHistoryButton();
        this.node.textPreviewWidget.parentEl.appendChild(button);
    };

    paintSaveHistoryButton = () => {
        if (!this.saveHistoryButton) return;
        const active = this.shouldSaveHistory();
        this.saveHistoryButton.textContent = active ? "Save" : "Temp";
        this.saveHistoryButton.style.opacity = active ? "1" : "0.65";
        this.saveHistoryButton.style.borderColor = active ? "rgba(95,220,255,0.45)" : "rgba(255,255,255,0.18)";
        this.saveHistoryButton.style.background = active ? "rgba(95,220,255,0.16)" : "rgba(0,0,0,0.35)";
    };

    updateHistoryLabel = () => {
        if (this.history.length > 0) {
            this.historyLabel.innerText = `${this.historyIndex + 1}/${this.history.length}`;
        } else {
            this.historyLabel.innerText = "-/-";
        }
    };

    navigateHistory = (direction) => {
        if (this.history.length === 0) return;
        const newIndex = this.historyIndex + direction;
        if (newIndex >= 0 && newIndex < this.history.length) {
            this.historyIndex = newIndex;
            this.refreshText();
        }
    };

    addTextToHistory = (text) => {
        const value = String(text);
        this.history.push(value);
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        }
        this.historyIndex = this.history.length - 1;
        this.refreshText();
    };

    refreshText = ({ syncHistory = true } = {}) => {
        if (!this.textareaEl) return;
        if (this.historyIndex >= 0 && this.historyIndex < this.history.length) {
            this.textareaEl.value = this.history[this.historyIndex];
        } else {
            this.textareaEl.value = "";
        }
        this.updateHistoryLabel();
        this.syncStoredTextWidget(this.textareaEl.value);
        if (syncHistory) this.syncHistoryWidget();
    };
}
