import { app } from "/scripts/app.js";

function findWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
    if (!widget || widget._tkswHidden) return;
    widget._tkswHidden = true;
    widget.serialize = true;
    widget.type = "hidden";
    widget.hidden = true;
    widget.computeSize = () => [0, 0];
    widget.draw = () => {};
    if (widget.element?.style) widget.element.style.display = "none";
}

function chainCallback(object, property, callback) {
    const original = object?.[property];
    object[property] = function () {
        const result = original?.apply(this, arguments);
        callback?.apply(this, arguments);
        return result;
    };
}

function applyFlexLayout(node, domWidget) {
    if (node._tkswElementalFlex) return;
    try {
        const widgetElement = domWidget.element;
        const widgetContainer = widgetElement.parentElement;
        const widgetsArea = widgetContainer.parentElement;
        const contentArea = widgetsArea.parentElement;
        contentArea.style.display = "flex";
        contentArea.style.flexDirection = "column";
        contentArea.style.overflow = "hidden";
        widgetContainer.style.flexGrow = "1";
        widgetContainer.style.minHeight = "0";
        widgetContainer.style.height = "100%";
        widgetContainer.style.overflow = "hidden";
        node._tkswElementalFlex = true;
    } catch (error) {
        console.error("[tksw_node] Failed to apply Elemental UI layout:", error);
    }
}

function defaultState() {
    return {
        items: [
            { pattern: "lora_unet_input_blocks_4", strength: 0.5, enabled: true },
        ],
    };
}

function parseState(widget) {
    const state = defaultState();
    try {
        const parsed = JSON.parse(widget?.value || "{}");
        if (Array.isArray(parsed.items)) state.items = parsed.items;
    } catch (_) {
        // Keep defaults.
    }
    state.items = state.items.map((item) => ({
        pattern: String(item?.pattern ?? ""),
        strength: Number.isFinite(Number(item?.strength)) ? Math.max(-2, Math.min(2, Number(item.strength))) : 1,
        enabled: item?.enabled !== false,
    }));
    if (!state.items.length) state.items.push({ pattern: "", strength: 1, enabled: true });
    return state;
}

function writeState(node, widget, state) {
    widget.value = JSON.stringify(state);
    widget.callback?.(widget.value);
    node.setDirtyCanvas?.(true, true);
}

function range(start, end) {
    return Array.from({ length: end - start + 1 }, (_, index) => start + index);
}

const ELEMENTAL_PRESETS = {
    "SDXL Attention": [
        "lora_te1_",
        "lora_te2_",
        ...[4, 5, 7, 8].map((n) => `lora_unet_input_blocks_${n}`),
        "lora_unet_middle_block",
        ...range(0, 5).map((n) => `lora_unet_output_blocks_${n}`),
    ],
    "SDXL Full": [
        "lora_te1_",
        "lora_te2_",
        ...range(0, 11).map((n) => `lora_unet_input_blocks_${n}`),
        "lora_unet_middle_block",
        ...range(0, 11).map((n) => `lora_unet_output_blocks_${n}`),
    ],
    "FLUX": [
        ...range(0, 18).map((n) => `double_${n}`),
        ...range(0, 37).map((n) => `single_${n}`),
    ],
    "Z-Image": range(0, 29).map((n) => `diffusion_model.layers.${n}`),
    "Wan": range(0, 39).map((n) => `blocks.${n}`),
    "Qwen": range(0, 59).map((n) => `transformer_blocks.${n}`),
    "Anima": [
        ...range(0, 27).map((n) => `diffusion_model.blocks.${n}`),
        ...range(0, 5).map((n) => `diffusion_model.llm_adapter.blocks.${n}`),
    ],
};

class ElementalRuleEditor {
    constructor(node) {
        this.node = node;
        this.settingsWidget = findWidget(node, "elemental_settings_json");
        this.state = parseState(this.settingsWidget);
        this.rows = [];
        this.viewMode = "list";
        this.graphEditMode = "single";
        this.graphSnapMode = "free";
        this.graphItemWidth = 18;
        this.resizeFrame = null;
        this.tooltip = null;
        this.graphScale = null;
        this.graphGuide = null;
        this.graphDrawDragging = false;
        this.graphDrawLast = null;
        this.graphDrawToggleDragging = false;
        this.graphDrawToggleValue = null;

        this.root = document.createElement("div");
        Object.assign(this.root.style, {
            display: "flex",
            flexDirection: "column",
            gap: "6px",
            padding: "6px",
            height: "100%",
            minHeight: "0",
            boxSizing: "border-box",
            overflow: "hidden",
            background: "rgba(0,0,0,0.28)",
            borderRadius: "4px",
            color: "var(--fg-color)",
        });
        this.root.appendChild(this.buildToolbar());
        this.root.appendChild(this.buildPresetBar());
        this.root.appendChild(this.buildModeBar());
        this.list = document.createElement("div");
        Object.assign(this.list.style, {
            display: "flex",
            flexDirection: "column",
            gap: "3px",
            flex: "1 1 auto",
            minHeight: "0",
            overflow: "auto",
            paddingRight: "3px",
        });
        this.root.appendChild(this.list);
        this.tooltip = document.createElement("div");
        Object.assign(this.tooltip.style, {
            position: "fixed",
            zIndex: "9999",
            display: "none",
            maxWidth: "280px",
            padding: "4px 6px",
            border: "1px solid rgba(255,255,255,0.18)",
            background: "rgba(10,12,16,0.94)",
            color: "var(--fg-color)",
            fontSize: "10px",
            lineHeight: "13px",
            pointerEvents: "none",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
            boxShadow: "0 2px 8px rgba(0,0,0,0.35)",
        });
        document.body.appendChild(this.tooltip);
        this.rebuild();
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(() => this.scheduleLayout());
            this.resizeObserver.observe(this.root);
            this.resizeObserver.observe(this.list);
        }
    }

    hideTooltip() {
        if (this.tooltip) this.tooltip.style.display = "none";
    }

    showTooltip(event, item, index) {
        if (!this.tooltip || this.viewMode === "list") return;
        const pattern = String(item.pattern || `#${index + 1}`);
        const enabled = item.enabled ? "ON" : "OFF";
        this.tooltip.textContent = `${this.shortLabel(pattern, index)}  ${Number(item.strength).toFixed(2)}  ${enabled}  ${pattern}`;
        this.tooltip.style.display = "block";
        const margin = 10;
        const rect = this.tooltip.getBoundingClientRect();
        let left = event.clientX + 12;
        if (left + rect.width + margin > window.innerWidth) left = event.clientX - rect.width - 12;
        left = Math.max(margin, Math.min(left, window.innerWidth - rect.width - margin));
        let top = event.clientY - rect.height - 10;
        if (top < margin) top = event.clientY + 14;
        top = Math.max(margin, Math.min(top, window.innerHeight - rect.height - margin));
        this.tooltip.style.left = `${left}px`;
        this.tooltip.style.top = `${top}px`;
    }

    button(label, title, action) {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = label;
        button.title = title;
        button.onclick = action;
        Object.assign(button.style, {
            display: "inline-flex",
            alignItems: "center",
            justifyContent: "center",
            height: "20px",
            minWidth: "32px",
            padding: "3px 5px 1px",
            border: "1px solid rgba(255,255,255,0.16)",
            borderRadius: "0",
            background: "rgba(0,0,0,0.16)",
            color: "var(--fg-color)",
            lineHeight: "12px",
            boxSizing: "border-box",
        });
        button.addEventListener("mouseenter", () => button.style.background = "rgba(255,255,255,0.07)");
        button.addEventListener("mouseleave", () => button.style.background = "rgba(0,0,0,0.16)");
        button.addEventListener("pointerdown", () => button.style.background = "rgba(255,255,255,0.11)");
        button.addEventListener("pointerup", () => button.style.background = "rgba(255,255,255,0.07)");
        return button;
    }

    buildToolbar() {
        const bar = document.createElement("div");
        Object.assign(bar.style, {
            display: "grid",
            gridTemplateColumns: "1fr repeat(8, auto)",
            gap: "4px",
            alignItems: "center",
        });
        const hint = document.createElement("span");
        hint.textContent = "Elemental rules";
        Object.assign(hint.style, {
            fontSize: "10px",
            opacity: "0.8",
            whiteSpace: "nowrap",
            overflow: "hidden",
            textOverflow: "ellipsis",
        });
        bar.appendChild(hint);
        bar.appendChild(this.button("0", "Set all strengths to 0", () => this.setAllStrengths(0)));
        bar.appendChild(this.button(".5", "Set all strengths to 0.5", () => this.setAllStrengths(0.5)));
        bar.appendChild(this.button("1", "Set all strengths to 1.0", () => this.setAllStrengths(1)));
        bar.appendChild(this.button("+", "Add rule", () => this.addRule()));
        bar.appendChild(this.button("On", "Enable all rules", () => this.setAllEnabled(true)));
        bar.appendChild(this.button("Off", "Disable all rules", () => this.setAllEnabled(false)));
        bar.appendChild(this.button("Import", "Import from text lines", () => this.importText()));
        bar.appendChild(this.button("Copy", "Copy enabled rules as text", () => this.copyText()));
        return bar;
    }

    buildPresetBar() {
        const bar = document.createElement("div");
        Object.assign(bar.style, {
            display: "grid",
            gridTemplateColumns: "auto minmax(0, 1fr) auto",
            gap: "4px",
            alignItems: "center",
        });
        const label = document.createElement("span");
        label.textContent = "Preset";
        Object.assign(label.style, {
            fontSize: "10px",
            opacity: "0.8",
        });
        this.presetSelect = document.createElement("select");
        Object.assign(this.presetSelect.style, {
            width: "100%",
            minWidth: "0",
            height: "20px",
            boxSizing: "border-box",
        });
        const placeholder = document.createElement("option");
        placeholder.value = "";
        placeholder.textContent = "Choose model rule set...";
        this.presetSelect.appendChild(placeholder);
        for (const name of Object.keys(ELEMENTAL_PRESETS)) {
            const option = document.createElement("option");
            option.value = name;
            option.textContent = name;
            this.presetSelect.appendChild(option);
        }
        bar.appendChild(label);
        bar.appendChild(this.presetSelect);
        bar.appendChild(this.button("Apply", "Replace rules with selected preset", () => this.applyPreset()));
        return bar;
    }

    buildModeBar() {
        const bar = document.createElement("div");
        Object.assign(bar.style, {
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr auto auto",
            gap: "4px",
            alignItems: "center",
        });
        this.listButton = this.button("List", "Edit patterns and strengths in rows", () => this.setViewMode("list"));
        this.eqButton = this.button("EQ", "Edit strengths as vertical sliders", () => this.setViewMode("equalizer"));
        this.graphButton = this.button("Graph", "Edit strengths as compact graph bars", () => this.setViewMode("graph"));
        this.drawButton = this.button("Draw", "Toggle graph draw-across-items editing", () => this.toggleGraphDrawMode());
        this.snapButton = this.button("Snap", "Toggle graph snap to 5% guide lines", () => this.toggleGraphSnapMode());
        this.drawButton.style.width = "44px";
        this.snapButton.style.width = "44px";
        bar.appendChild(this.listButton);
        bar.appendChild(this.eqButton);
        bar.appendChild(this.graphButton);
        bar.appendChild(this.drawButton);
        bar.appendChild(this.snapButton);
        this.paintModeButtons();
        return bar;
    }

    setViewMode(mode) {
        if (this.viewMode === mode) return;
        this.viewMode = mode;
        this.paintModeButtons();
        this.rebuild();
    }

    toggleGraphDrawMode() {
        this.graphEditMode = this.graphEditMode === "draw" ? "single" : "draw";
        this.paintModeButtons();
        this.rebuild();
    }

    toggleGraphSnapMode() {
        this.graphSnapMode = this.graphSnapMode === "guide" ? "free" : "guide";
        this.paintModeButtons();
    }

    paintButton(button, active, visible = true) {
        if (!button) return;
        button.style.display = visible ? "" : "none";
        button.style.opacity = active ? "1" : "0.62";
        button.style.fontWeight = "600";
        button.style.borderColor = active ? "rgba(95,220,255,0.55)" : "rgba(255,255,255,0.16)";
        button.style.background = active ? "rgba(95,220,255,0.18)" : "rgba(0,0,0,0.16)";
    }

    paintModeButtons() {
        this.paintButton(this.listButton, this.viewMode === "list");
        this.paintButton(this.eqButton, this.viewMode === "equalizer");
        this.paintButton(this.graphButton, this.viewMode === "graph");
        const graph = this.viewMode === "graph";
        this.paintButton(this.drawButton, this.graphEditMode === "draw", graph);
        this.paintButton(this.snapButton, this.graphSnapMode === "guide", graph);
    }

    rebuild() {
        this.rows = [];
        this.list.replaceChildren();
        this.applyListLayout();
        if (this.viewMode === "graph") {
            this.graphScale = this.buildGraphScale();
            this.graphGuide = this.buildGraphGuide();
            this.list.appendChild(this.graphScale);
            this.list.appendChild(this.graphGuide);
        } else {
            this.graphScale = null;
            this.graphGuide = null;
        }
        for (let index = 0; index < this.state.items.length; index++) {
            this.list.appendChild(this.buildRow(index));
        }
        this.scheduleLayout();
    }

    applyListLayout() {
        const graph = this.viewMode === "graph";
        Object.assign(this.list.style, {
            position: "relative",
            display: graph || this.viewMode === "equalizer" ? "flex" : "flex",
            flexDirection: graph || this.viewMode === "equalizer" ? "row" : "column",
            alignItems: graph || this.viewMode === "equalizer" ? "stretch" : "stretch",
            gap: graph ? "0" : this.viewMode === "equalizer" ? "2px" : "3px",
            overflowX: graph || this.viewMode === "equalizer" ? "auto" : "hidden",
            overflowY: graph ? "hidden" : "auto",
            paddingRight: "3px",
            paddingLeft: graph ? "30px" : "0",
        });
    }

    scheduleLayout() {
        if (this.resizeFrame) cancelAnimationFrame(this.resizeFrame);
        this.resizeFrame = requestAnimationFrame(() => this.updateLayout());
    }

    updateLayout() {
        this.resizeFrame = null;
        if (this.viewMode !== "graph") return;
        const count = Math.max(1, this.state.items.length);
        const scaleWidth = 30;
        const available = Math.max(0, this.list.clientWidth - scaleWidth - 8);
        this.graphItemWidth = Math.max(18, Math.floor(available / count));
        const rowHeight = Math.max(154, Math.floor(this.list.clientHeight || 0) - 4);
        this.graphRowHeight = rowHeight;
        for (const row of this.rows) row.applyLayout?.();
        const trackTop = 15;
        const trackHeight = Math.max(1, rowHeight - 28);
        if (this.graphScale) {
            Object.assign(this.graphScale.style, {
                display: "block",
                top: `${trackTop}px`,
                height: `${trackHeight}px`,
            });
            this.updateGraphScaleLabels();
        }
        if (this.graphGuide) {
            Object.assign(this.graphGuide.style, {
                display: "block",
                left: `${scaleWidth}px`,
                right: "3px",
                top: `${trackTop}px`,
                height: `${trackHeight}px`,
            });
        }
    }

    updateGraphScaleLabels() {
        const labels = ["2", "1", "0", "-1", "-2"];
        if (!this.graphScale) return;
        for (let index = 0; index < this.graphScale.children.length; index++) {
            this.graphScale.children[index].textContent = labels[index] ?? "";
        }
    }

    buildGraphScale() {
        const scale = document.createElement("div");
        Object.assign(scale.style, {
            display: "none",
            position: "absolute",
            left: "0",
            width: "28px",
            minWidth: "28px",
            fontSize: "8px",
            opacity: "0.7",
            textAlign: "right",
            paddingRight: "2px",
            boxSizing: "border-box",
            pointerEvents: "none",
            overflow: "visible",
            zIndex: "2",
        });
        for (let i = 0; i < 5; i++) {
            const item = document.createElement("span");
            Object.assign(item.style, {
                position: "absolute",
                right: "2px",
                top: `${i * 25}%`,
                transform: "translateY(-50%)",
                lineHeight: "9px",
            });
            scale.appendChild(item);
        }
        return scale;
    }

    buildGraphGuide() {
        const guide = document.createElement("div");
        Object.assign(guide.style, {
            display: "none",
            position: "absolute",
            pointerEvents: "none",
            overflow: "visible",
            zIndex: "0",
        });
        for (let i = 0; i < 21; i++) {
            const line = document.createElement("div");
            const edge = i === 0 || i === 20;
            const center = i === 10;
            const major = i % 5 === 0;
            const lineTop = i === 20 ? "calc(100% - 1px)" : `${i * 5}%`;
            Object.assign(line.style, {
                position: "absolute",
                left: "0",
                right: "0",
                top: lineTop,
                height: "1px",
                background: center
                    ? "rgba(255,255,255,0.18)"
                    : edge
                        ? "rgba(255,255,255,0.14)"
                        : major
                            ? "rgba(255,255,255,0.09)"
                            : "rgba(255,255,255,0.045)",
            });
            guide.appendChild(line);
        }
        return guide;
    }

    buildRow(index) {
        const item = this.state.items[index];
        const row = document.createElement("div");
        Object.assign(row.style, {
            display: "grid",
            gridTemplateColumns: "14px minmax(88px, 210px) minmax(110px, 1fr) 40px 22px",
            columnGap: "3px",
            rowGap: "2px",
            alignItems: "center",
            minHeight: "22px",
            padding: "2px 3px",
            border: "1px solid rgba(128,128,128,0.35)",
            borderRadius: "3px",
            boxSizing: "border-box",
        });

        const enabled = document.createElement("input");
        enabled.type = "checkbox";
        enabled.checked = item.enabled;
        enabled.style.margin = "0";

        const pattern = document.createElement("input");
        pattern.type = "text";
        pattern.value = item.pattern;
        pattern.placeholder = "lora_unet_input_blocks_4";
        pattern.title = "Pattern. match_mode controls prefix, contains, or regex matching.";
        Object.assign(pattern.style, {
            width: "100%",
            minWidth: "0",
            boxSizing: "border-box",
            fontSize: "10px",
        });

        const slider = document.createElement("input");
        slider.type = "range";
        slider.min = "-2";
        slider.max = "2";
        slider.step = "0.01";
        slider.value = String(item.strength);
        slider.style.width = "100%";

        const value = document.createElement("span");
        value.textContent = Number(item.strength).toFixed(2);
        Object.assign(value.style, {
            width: "42px",
            boxSizing: "border-box",
            fontSize: "10px",
            textAlign: "center",
            fontVariantNumeric: "tabular-nums",
        });

        const remove = this.button("x", "Delete rule", () => {
            this.state.items.splice(index, 1);
            if (!this.state.items.length) this.state.items.push({ pattern: "", strength: 1, enabled: true });
            this.save();
            this.rebuild();
        });
        remove.style.minWidth = "22px";
        remove.style.width = "22px";

        const sync = () => {
            item.enabled = enabled.checked;
            item.pattern = pattern.value;
            item.strength = Math.max(-2, Math.min(2, Number(slider.value)));
            value.textContent = item.strength.toFixed(2);
            row.style.opacity = item.enabled ? "1" : "0.45";
            paintGraph();
            this.save();
        };
        const adjustByWheel = (event) => {
            event.preventDefault();
            event.stopPropagation();
            const direction = (event.deltaY || event.deltaX) < 0 ? 1 : -1;
            const step = event.shiftKey ? 0.25 : event.altKey ? 0.01 : 0.05;
            slider.value = String(Math.max(-2, Math.min(2, Number(slider.value) + direction * step)));
            sync();
            this.showTooltip(event, item, index);
        };

        const label = document.createElement("div");
        label.textContent = this.shortLabel(item.pattern, index);
        label.title = item.pattern;
        Object.assign(label.style, {
            display: "none",
            fontSize: "8px",
            lineHeight: "9px",
            height: "9px",
            overflow: "hidden",
            textAlign: "center",
            whiteSpace: "nowrap",
            opacity: "0.75",
        });

        const toggle = document.createElement("span");
        Object.assign(toggle.style, {
            display: "none",
            width: "100%",
            height: "10px",
            background: item.enabled ? "rgba(255,255,255,0.72)" : "rgba(255,255,255,0.08)",
            cursor: "pointer",
        });
        toggle.onclick = (event) => {
            if (this.graphEditMode === "draw") return;
            event.preventDefault();
            event.stopPropagation();
            enabled.checked = !enabled.checked;
            sync();
            this.showTooltip(event, item, index);
        };

        const track = document.createElement("div");
        Object.assign(track.style, {
            display: "none",
            position: "relative",
            height: "auto",
            minHeight: "112px",
            background: "transparent",
            cursor: "ns-resize",
            overflow: "hidden",
        });
        const bar = document.createElement("div");
        Object.assign(bar.style, {
            position: "absolute",
            left: "-1px",
            right: "-1px",
            top: "50%",
            bottom: "50%",
            height: "auto",
            background: "rgba(75,190,255,0.22)",
            zIndex: "1",
        });
        const line = document.createElement("div");
        Object.assign(line.style, {
            position: "absolute",
            left: "-1px",
            right: "-1px",
            top: "50%",
            height: "1px",
            background: "rgb(95,220,255)",
            zIndex: "2",
        });
        track.appendChild(bar);
        track.appendChild(line);

        const valuePercent = () => (Math.max(-2, Math.min(2, Number(slider.value))) + 2) / 4;
        const paintGraph = () => {
            const percent = valuePercent() * 100;
            const lineTop = Math.max(0, Math.min(100, 100 - percent));
            if (percent >= 50) {
                bar.style.top = `${lineTop}%`;
                bar.style.bottom = "50%";
            } else {
                bar.style.top = "50%";
                bar.style.bottom = `${percent}%`;
            }
            bar.style.height = "auto";
            line.style.top = `calc(${lineTop}% - ${lineTop === 100 ? 1 : 0}px)`;
            toggle.style.background = enabled.checked ? "rgba(255,255,255,0.72)" : "rgba(255,255,255,0.08)";
            row.style.opacity = enabled.checked ? "1" : "0.45";
            label.textContent = this.shortLabel(item.pattern, index);
            label.title = item.pattern;
        };
        const setFromClientY = (clientY, event = null) => {
            const rect = track.getBoundingClientRect();
            if (rect.height <= 0) return;
            let percent = Math.max(0, Math.min(1, 1 - ((clientY - rect.top) / rect.height)));
            if (this.graphSnapMode === "guide") percent = Math.round(percent * 20) / 20;
            slider.value = String(-2 + percent * 4);
            sync();
            if (event) this.showTooltip(event, item, index);
        };
        const pointerDown = (event) => {
            event.preventDefault();
            if (this.graphEditMode === "draw") {
                this.startGraphDraw(event);
                return;
            }
            track.setPointerCapture?.(event.pointerId);
            setFromClientY(event.clientY, event);
            const move = (moveEvent) => setFromClientY(moveEvent.clientY, moveEvent);
            const up = () => {
                window.removeEventListener("pointermove", move);
                window.removeEventListener("pointerup", up);
                window.removeEventListener("pointercancel", up);
                window.removeEventListener("blur", up);
                track.removeEventListener?.("lostpointercapture", up);
            };
            window.addEventListener("pointermove", move);
            window.addEventListener("pointerup", up, { once: true });
            window.addEventListener("pointercancel", up, { once: true });
            window.addEventListener("blur", up, { once: true });
            track.addEventListener?.("lostpointercapture", up, { once: true });
        };
        track.onpointerdown = pointerDown;
        toggle.onpointerdown = (event) => {
            if (this.graphEditMode !== "draw") return;
            this.startGraphDrawToggle(event, !enabled.checked);
        };

        const rowApi = { applyLayout: null, track, rowElement: row, enabled, sync, setFromClientY, item, index };
        row._tkswElementalRow = rowApi;
        row._tkswElementalSetFromY = setFromClientY;
        track._tkswElementalRow = rowApi;
        toggle._tkswElementalRow = rowApi;

        enabled.onchange = sync;
        pattern.oninput = sync;
        slider.oninput = sync;
        row.onwheel = adjustByWheel;
        row.onpointermove = (event) => this.showTooltip(event, item, index);
        row.onpointerleave = () => this.hideTooltip();
        row.style.opacity = item.enabled ? "1" : "0.45";

        row.appendChild(enabled);
        row.appendChild(pattern);
        row.appendChild(slider);
        row.appendChild(value);
        row.appendChild(remove);
        row.appendChild(toggle);
        row.appendChild(track);
        row.appendChild(label);
        const applyLayout = () => {
            if (this.viewMode === "equalizer") {
                Object.assign(row.style, {
                    display: "grid",
                    gridTemplateColumns: "30px",
                    gridTemplateRows: "14px 16px minmax(118px, 1fr) 18px",
                    justifyItems: "center",
                    alignItems: "center",
                    gap: "1px",
                    width: "30px",
                    minWidth: "30px",
                    height: "100%",
                    minHeight: "154px",
                    padding: "3px 2px",
                });
                enabled.style.display = "";
                enabled.style.gridRow = "1";
                enabled.style.gridColumn = "1";
                enabled.style.justifySelf = "center";
                enabled.style.transform = "translateX(-3px)";
                label.style.display = "block";
                label.style.width = "30px";
                label.style.gridRow = "2";
                label.style.gridColumn = "1";
                label.style.justifySelf = "center";
                label.style.transform = "translateX(-3px)";
                slider.style.display = "";
                slider.style.width = "22px";
                slider.style.height = "100%";
                slider.style.minHeight = "112px";
                slider.style.writingMode = "vertical-lr";
                slider.style.direction = "rtl";
                slider.style.gridRow = "3";
                slider.style.gridColumn = "1";
                slider.style.justifySelf = "center";
                slider.style.transform = "translateX(-3px)";
                value.style.display = "";
                value.style.width = "28px";
                value.style.gridRow = "4";
                value.style.gridColumn = "1";
                value.style.justifySelf = "center";
                value.style.transform = "translateX(-3px)";
                pattern.style.display = "none";
                pattern.style.gridRow = "";
                pattern.style.gridColumn = "";
                remove.style.display = "none";
                remove.style.gridRow = "";
                remove.style.gridColumn = "";
                toggle.style.display = "none";
                toggle.style.gridRow = "";
                toggle.style.gridColumn = "";
                track.style.display = "none";
                track.style.gridRow = "";
                track.style.gridColumn = "";
            } else if (this.viewMode === "graph") {
                const width = this.graphItemWidth || 18;
                const rowHeight = this.graphRowHeight || Math.max(154, Math.floor(this.list.clientHeight || 0) - 4);
                Object.assign(row.style, {
                    display: "grid",
                    gridTemplateColumns: `${width}px`,
                    gridTemplateRows: "12px minmax(118px, 1fr) 10px",
                    justifyItems: "stretch",
                    alignItems: "stretch",
                    gap: "1px",
                    width: `${width}px`,
                    minWidth: "18px",
                    height: `${rowHeight}px`,
                    minHeight: "154px",
                    padding: "2px 0",
                    border: "0",
                    borderRight: "1px solid rgba(255,255,255,0.018)",
                    borderRadius: "0",
                    position: "relative",
                    zIndex: "1",
                });
                enabled.style.display = "none";
                enabled.style.gridRow = "";
                enabled.style.gridColumn = "";
                enabled.style.justifySelf = "";
                enabled.style.transform = "";
                pattern.style.display = "none";
                pattern.style.gridRow = "";
                pattern.style.gridColumn = "";
                slider.style.display = "none";
                slider.style.gridRow = "";
                slider.style.gridColumn = "";
                slider.style.justifySelf = "";
                slider.style.transform = "";
                value.style.display = "none";
                value.style.gridRow = "";
                value.style.gridColumn = "";
                value.style.justifySelf = "";
                value.style.transform = "";
                remove.style.display = "none";
                remove.style.gridRow = "";
                remove.style.gridColumn = "";
                toggle.style.display = "block";
                toggle.style.gridRow = "1";
                toggle.style.gridColumn = "1";
                track.style.display = "block";
                track.style.gridRow = "2";
                track.style.gridColumn = "1";
                track.style.height = "100%";
                track.style.minHeight = "0";
                track.style.cursor = this.graphEditMode === "draw" ? "crosshair" : "ns-resize";
                label.style.display = "block";
                label.style.gridRow = "3";
                label.style.gridColumn = "1";
                label.style.justifySelf = "";
                label.style.transform = "";
            } else {
                Object.assign(row.style, {
                    display: "grid",
                    gridTemplateColumns: "14px minmax(88px, 210px) minmax(110px, 1fr) 40px 22px",
                    columnGap: "3px",
                    rowGap: "2px",
                    alignItems: "center",
                    minHeight: "22px",
                    padding: "2px 3px",
                    border: "1px solid rgba(128,128,128,0.35)",
                    borderRight: "",
                    borderRadius: "3px",
                    width: "",
                    minWidth: "",
                    height: "",
                });
                enabled.style.display = "";
                enabled.style.gridRow = "";
                enabled.style.gridColumn = "";
                enabled.style.justifySelf = "";
                enabled.style.transform = "";
                pattern.style.display = "";
                pattern.style.gridRow = "";
                pattern.style.gridColumn = "";
                slider.style.display = "";
                slider.style.width = "100%";
                slider.style.height = "";
                slider.style.minHeight = "";
                slider.style.writingMode = "";
                slider.style.direction = "";
                slider.style.gridRow = "";
                slider.style.gridColumn = "";
                slider.style.justifySelf = "";
                slider.style.transform = "";
                value.style.display = "";
                value.style.width = "40px";
                value.style.gridRow = "";
                value.style.gridColumn = "";
                value.style.justifySelf = "";
                value.style.transform = "";
                remove.style.display = "";
                remove.style.gridRow = "";
                remove.style.gridColumn = "";
                toggle.style.display = "none";
                toggle.style.gridRow = "";
                toggle.style.gridColumn = "";
                track.style.display = "none";
                track.style.gridRow = "";
                track.style.gridColumn = "";
                label.style.display = "none";
                label.style.gridRow = "";
                label.style.gridColumn = "";
                label.style.justifySelf = "";
                label.style.transform = "";
            }
            paintGraph();
        };
        applyLayout();
        rowApi.applyLayout = applyLayout;
        this.rows.push(rowApi);
        return row;
    }

    graphRowAtPointer(event) {
        let element = document.elementFromPoint(event.clientX, event.clientY);
        while (element && !element._tkswElementalRow) element = element.parentElement;
        if (element?._tkswElementalRow) return element._tkswElementalRow;
        for (const row of this.rows) {
            const rect = row.rowElement?.getBoundingClientRect?.();
            if (!rect) continue;
            if (event.clientX >= rect.left && event.clientX <= rect.right) return row;
        }
        return null;
    }

    startGraphDraw(event) {
        event.preventDefault();
        event.currentTarget?.setPointerCapture?.(event.pointerId);
        this.graphDrawDragging = true;
        this.graphDrawLast = null;
        const update = (moveEvent) => {
            const row = this.graphRowAtPointer(moveEvent);
            if (!row) return;
            const currentIndex = this.rows.indexOf(row);
            if (this.graphDrawLast && currentIndex >= 0) {
                const previousIndex = this.graphDrawLast.index;
                const start = Math.min(previousIndex, currentIndex);
                const end = Math.max(previousIndex, currentIndex);
                const span = currentIndex - previousIndex;
                if (span !== 0) {
                    for (let index = start; index <= end; index++) {
                        const target = this.rows[index];
                        if (!target) continue;
                        const t = (index - previousIndex) / span;
                        const y = this.graphDrawLast.clientY + (moveEvent.clientY - this.graphDrawLast.clientY) * t;
                        target.setFromClientY(y, moveEvent);
                    }
                } else {
                    row.setFromClientY(moveEvent.clientY, moveEvent);
                }
            } else {
                row.setFromClientY(moveEvent.clientY, moveEvent);
            }
            this.graphDrawLast = { index: currentIndex, clientY: moveEvent.clientY };
        };
        update(event);
        const move = (moveEvent) => {
            if (this.graphDrawDragging) update(moveEvent);
        };
        const up = () => {
            this.graphDrawDragging = false;
            this.graphDrawLast = null;
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            window.removeEventListener("pointercancel", up);
            window.removeEventListener("blur", up);
            event.currentTarget?.removeEventListener?.("lostpointercapture", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up, { once: true });
        window.addEventListener("pointercancel", up, { once: true });
        window.addEventListener("blur", up, { once: true });
        event.currentTarget?.addEventListener?.("lostpointercapture", up, { once: true });
    }

    updateGraphDrawToggleAtPointer(event) {
        const row = this.graphRowAtPointer(event);
        if (!row || row.enabled.checked === this.graphDrawToggleValue) return;
        row.enabled.checked = this.graphDrawToggleValue;
        row.sync();
        this.showTooltip(event, row.item, row.index);
    }

    startGraphDrawToggle(event, initialValue) {
        event.preventDefault();
        event.stopPropagation();
        event.currentTarget?.setPointerCapture?.(event.pointerId);
        this.graphDrawToggleDragging = true;
        this.graphDrawToggleValue = initialValue;
        this.updateGraphDrawToggleAtPointer(event);
        const move = (moveEvent) => {
            if (this.graphDrawToggleDragging) this.updateGraphDrawToggleAtPointer(moveEvent);
        };
        const up = () => {
            this.graphDrawToggleDragging = false;
            this.graphDrawToggleValue = null;
            window.removeEventListener("pointermove", move);
            window.removeEventListener("pointerup", up);
            window.removeEventListener("pointercancel", up);
            window.removeEventListener("blur", up);
            event.currentTarget?.removeEventListener?.("lostpointercapture", up);
        };
        window.addEventListener("pointermove", move);
        window.addEventListener("pointerup", up, { once: true });
        window.addEventListener("pointercancel", up, { once: true });
        window.addEventListener("blur", up, { once: true });
        event.currentTarget?.addEventListener?.("lostpointercapture", up, { once: true });
    }

    shortLabel(pattern, index) {
        const text = String(pattern || `#${index + 1}`);
        const lower = text.toLowerCase();
        let match;
        if (lower.includes("te1")) return "te1";
        if (lower.includes("te2")) return "te2";
        if (lower.includes("middle_block") || lower.includes("unet_mid")) return "mid";
        match = lower.match(/input_blocks?[_\.-]?(\d+)/);
        if (match) return `i${match[1]}`;
        match = lower.match(/output_blocks?[_\.-]?(\d+)/);
        if (match) return `o${match[1]}`;
        match = lower.match(/double[_\.-]?(\d+)/);
        if (match) return `d${match[1]}`;
        match = lower.match(/single[_\.-]?(\d+)/);
        if (match) return `s${match[1]}`;
        match = lower.match(/llm_adapter\.blocks\.(\d+)/);
        if (match) return `la${match[1]}`;
        match = lower.match(/diffusion_model\.blocks\.(\d+)/);
        if (match) return `b${match[1]}`;
        match = lower.match(/diffusion_model\.layers\.(\d+)/);
        if (match) return `l${match[1]}`;
        match = lower.match(/transformer_blocks\.(\d+)/);
        if (match) return `t${match[1]}`;
        match = lower.match(/blocks\.(\d+)/);
        if (match) return `b${match[1]}`;
        match = lower.match(/(?:block|blocks?|layers?)[_.-]?(\d+)/);
        if (match) return match[1];
        return text.replace(/[.*+?^${}()|[\]\\]/g, "").slice(0, 5) || `#${index + 1}`;
    }

    addRule() {
        this.state.items.push({ pattern: "", strength: 1, enabled: true });
        this.save();
        this.rebuild();
    }

    setAllEnabled(value) {
        for (const item of this.state.items) item.enabled = value;
        this.save();
        this.rebuild();
    }

    setAllStrengths(value) {
        for (const item of this.state.items) item.strength = value;
        this.save();
        this.rebuild();
    }

    applyPreset() {
        const name = this.presetSelect?.value;
        const patterns = ELEMENTAL_PRESETS[name];
        if (!patterns) return;
        this.state.items = patterns.map((pattern) => ({
            pattern,
            strength: 1,
            enabled: true,
        }));
        this.save();
        this.rebuild();
    }

    importText() {
        const text = prompt("Paste one rule per line. Prefix with # to disable.", this.toText(true));
        if (text === null) return;
        const items = [];
        for (const rawLine of text.split(/\r?\n/)) {
            let line = rawLine.trim();
            if (!line) continue;
            let enabled = true;
            if (line.startsWith("#")) {
                enabled = false;
                line = line.slice(1).trim();
            }
            if (!line.includes("=")) continue;
            const [pattern, strengthText] = line.split(/=(.*)/s);
            const strength = Number(strengthText.trim());
            items.push({
                pattern: pattern.trim(),
                strength: Number.isFinite(strength) ? strength : 1,
                enabled,
            });
        }
        this.state.items = items.length ? items : [{ pattern: "", strength: 1, enabled: true }];
        this.save();
        this.rebuild();
    }

    copyText() {
        const text = this.toText(true);
        navigator.clipboard?.writeText(text);
    }

    toText(includeDisabled = false) {
        return this.state.items
            .filter((item) => includeDisabled || item.enabled)
            .map((item) => `${item.enabled ? "" : "# "}${item.pattern} = ${Number(item.strength).toFixed(2)}`)
            .join("\n");
    }

    save() {
        writeState(this.node, this.settingsWidget, this.state);
    }

    reloadFromWidget() {
        this.state = parseState(this.settingsWidget);
        this.rebuild();
    }
}

function setupNode(node, nodeData) {
    if (node._tkswElementalEditor) return;
    const settingsWidget = findWidget(node, "elemental_settings_json");
    if (!settingsWidget) return;
    hideWidget(settingsWidget);

    const editor = new ElementalRuleEditor(node);
    node._tkswElementalEditor = editor;

    const element = document.createElement("div");
    element.style.width = "100%";
    element.style.height = "100%";
    element.style.minHeight = "0";
    element.style.overflow = "hidden";
    element.appendChild(editor.root);

    const domWidget = node.addDOMWidget("Elemental Rules", "TkswElementalRules", element, {
        serialize: false,
        getValue: () => element,
        setValue: () => {},
        hideOnZoom: false,
    });
    applyFlexLayout(node, domWidget);
    node.setSize([Math.max(node.size?.[0] ?? 320, 520), Math.max(node.size?.[1] ?? 0, 360)]);
}

app.registerExtension({
    name: "tksw_node.LoraLoaderElementalUI",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "LoraLoaderElementalUI") return;
        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            setupNode(this, nodeData);
        });
        chainCallback(nodeType.prototype, "onConfigure", function () {
            this._tkswElementalEditor?.reloadFromWidget();
        });
    },
});
