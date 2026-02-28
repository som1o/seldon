/*
 * SeldonGui.cpp — GTK4 desktop front-end for the Seldon analytics pipeline.
 *
 * Design rules:
 *  • Native Adwaita-dark look — no gradients, no glow, no glassmorphism.
 *  • Every path input is a dedicated "Choose…" button + a readable path label.
 *    Users NEVER have to type a path.
 *  • Zero deprecation-warnings: only GtkFileDialog (GTK >= 4.10) is used.
 *  • All AutoConfig fields are exposed across six tabs.
 *  • Run panel shows an Ubuntu-installer style progress overlay with a
 *    live progress bar, step label, and a terminal icon for the full log.
 */

#include "SeldonGui.h"
#include "AutomationPipeline.h"
#include "SeldonExceptions.h"

#include <gtk/gtk.h>

#include <atomic>
#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <unistd.h>

namespace {

/* ── TeeStreambuf ──────────────────────────────────────────────────────────
 * Installed over std::cout while the pipeline runs.
 * Writes every character to the original (real) stdout AND dispatches
 * complete lines (delimited by \n or \r) to the GUI main thread.         */

struct GuiState; // forward-declare for TeeStreambuf

struct LineFwd {
    GuiState*   state;
    std::string line;
    bool        isSpin; /* true = carriage-return terminated (spinner line) */
};

/* forward declaration — implemented after GuiState */
static void gui_handle_pipeline_line(GuiState*, const std::string&, bool isSpin);

struct TeeStreambuf : public std::streambuf {
    explicit TeeStreambuf(std::streambuf* real, GuiState* st)
        : real_(real), state_(st) {}

    int overflow(int c) override {
        if (c == EOF) return c;
        char ch = static_cast<char>(c);
        real_->sputc(ch);
        ingest(ch);
        return c;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        real_->sputn(s, n);
        for (std::streamsize i = 0; i < n; ++i) ingest(s[i]);
        return n;
    }

private:
    void ingest(char c) {
        if (c == '\r') {
            flush_line(true);
        } else if (c == '\n') {
            flush_line(false);
        } else {
            buf_.push_back(c);
        }
    }

    void flush_line(bool isSpin) {
        if (buf_.empty() && !isSpin) return;
        std::string line = std::move(buf_);
        buf_.clear();
        auto* d = new LineFwd{state_, std::move(line), isSpin};
        g_main_context_invoke(nullptr, [](gpointer p) -> gboolean {
            std::unique_ptr<LineFwd> dd(static_cast<LineFwd*>(p));
            gui_handle_pipeline_line(dd->state, dd->line, dd->isSpin);
            return G_SOURCE_REMOVE;
        }, d);
    }

    std::streambuf* real_;
    GuiState*       state_;
    std::string     buf_;
};

/* ── helpers ──────────────────────────────────────────────────────────────── */

std::string trim(std::string s) {
    auto notSpace = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), notSpace));
    s.erase(std::find_if(s.rbegin(), s.rend(), notSpace).base(), s.end());
    return s;
}

std::string bool_flag(bool v) { return v ? "true" : "false"; }

std::vector<std::string> split_args_inline(const std::string& text) {
    std::vector<std::string> tokens;
    std::string cur;
    bool inS = false, inD = false, esc = false;
    for (char c : text) {
        if (esc) { cur.push_back(c); esc = false; continue; }
        if (c == '\\') { esc = true; continue; }
        if (!inD && c == '\'') { inS = !inS; continue; }
        if (!inS && c == '"')  { inD = !inD; continue; }
        if (!inS && !inD && std::isspace(static_cast<unsigned char>(c))) {
            if (!cur.empty()) { tokens.push_back(cur); cur.clear(); }
            continue;
        }
        cur.push_back(c);
    }
    if (!cur.empty()) tokens.push_back(cur);
    return tokens;
}

std::string create_config_overlay(const std::string& configPath,
                                  const std::string& overrides,
                                  std::string& error) {
    const std::string tc = trim(configPath);
    const std::string to = trim(overrides);
    if (tc.empty() && to.empty()) return "";

    const auto tempPath = std::filesystem::temp_directory_path() /
                          ("seldon_gui_" + std::to_string(::getpid()) + ".yaml");
    std::ofstream out(tempPath);
    if (!out) { error = "Cannot create temp config file."; return ""; }

    if (!tc.empty()) {
        std::ifstream in(tc);
        if (!in) { error = "Cannot read base config: " + tc; return ""; }
        out << in.rdbuf() << "\n";
    }
    if (!to.empty()) { out << "\n# GUI overrides\n" << to << "\n"; }
    return tempPath.string();
}

/* ── state structs ────────────────────────────────────────────────────────── */

struct GuiState {
    GtkApplication* app    = nullptr;
    GtkWindow*      window = nullptr;

    /* path storage — hidden GtkEntry widgets; text read by build_args */
    GtkEntry* datasetEntry = nullptr;
    GtkEntry* outputEntry  = nullptr;
    GtkEntry* configEntry  = nullptr;

    /* editable text fields */
    GtkEntry*    targetEntry    = nullptr;
    GtkEntry*    assetsEntry    = nullptr;
    GtkEntry*    delimiterEntry = nullptr;
    GtkEntry*    excludeEntry   = nullptr;
    GtkEntry*    reportEntry    = nullptr;
    GtkDropDown* exportPreprocessedDrop = nullptr;

    /* analysis */
    GtkDropDown*   profileDrop           = nullptr;
    GtkDropDown*   outlierMethodDrop     = nullptr;
    GtkDropDown*   outlierActionDrop     = nullptr;
    GtkDropDown*   scalingDrop           = nullptr;
    GtkDropDown*   targetStrategyDrop    = nullptr;
    GtkDropDown*   featureStrategyDrop   = nullptr;
    GtkDropDown*   bivariateStrategyDrop = nullptr;
    GtkSwitch*     fastSwitch            = nullptr;
    GtkSwitch*     lowMemorySwitch       = nullptr;
    GtkSwitch*     verboseSwitch         = nullptr;
    GtkSpinButton* kfoldSpin             = nullptr;

    /* neural network */
    GtkDropDown*   neuralStrategyDrop      = nullptr;
    GtkDropDown*   neuralOptimizerDrop     = nullptr;
    GtkDropDown*   neuralOrdinalModeDrop   = nullptr;
    GtkDropDown*   neuralExplainDrop       = nullptr;
    GtkSpinButton* neuralLrSpin            = nullptr;
    GtkSpinButton* neuralMinLayersSpin     = nullptr;
    GtkSpinButton* neuralMaxLayersSpin     = nullptr;
    GtkSpinButton* neuralMaxHiddenSpin     = nullptr;
    GtkSwitch*     neuralStreamingSwitch   = nullptr;
    GtkSwitch*     neuralMultiOutputSwitch = nullptr;
    GtkSwitch*     neuralOodSwitch         = nullptr;

    /* feature engineering */
    GtkSwitch*     featPolySwitch  = nullptr;
    GtkSwitch*     featLogSwitch   = nullptr;
    GtkSwitch*     featRatioSwitch = nullptr;
    GtkSpinButton* featDegreeSpin  = nullptr;
    GtkSpinButton* featMaxBaseSpin = nullptr;

    /* plots */
    GtkSwitch*     plotUniSwitch     = nullptr;
    GtkSwitch*     plotOverallSwitch = nullptr;
    GtkSwitch*     plotBiSwitch      = nullptr;
    GtkSwitch*     htmlSwitch        = nullptr;
    GtkDropDown*   plotFormatDrop    = nullptr;
    GtkDropDown*   plotThemeDrop     = nullptr;
    GtkSwitch*     plotGridSwitch    = nullptr;
    GtkSpinButton* plotWidthSpin     = nullptr;
    GtkSpinButton* plotHeightSpin    = nullptr;
    GtkSpinButton* plotPointSizeSpin = nullptr;
    GtkSpinButton* plotLineWidthSpin = nullptr;

    /* advanced */
    GtkTextView* extraArgsView = nullptr;
    GtkTextView* overridesView = nullptr;

    /* ── bottom run-panel widgets ─────────────────────────────────────── */
    GtkLabel*    previewLabel    = nullptr; /* command preview label       */
    GtkButton*   runButton       = nullptr; /* header-bar run button       */

    /* idle status row */
    GtkWidget*   idleBar         = nullptr; /* row shown when not running  */
    GtkLabel*    statusLabel     = nullptr; /* "Ready." / "Completed."     */

    /* run progress overlay */
    GtkWidget*   progressPanel   = nullptr; /* replaces idleBar while running */
    GtkLabel*    stepLabel       = nullptr; /* "Analyzing bivariate pairs" */
    GtkLabel*    stepCountLabel  = nullptr; /* "Step 8 of 10"             */
    GtkProgressBar* progressBar  = nullptr; /* the big bar                */
    GtkLabel*    pctLabel        = nullptr; /* "80%"                      */
    GtkButton*   terminalBtn     = nullptr; /* opens log dialog           */

    /* ── shared log state (accessed from pipeline thread via GTK invoke) */
    std::mutex   logMutex;
    std::string  capturedLog;              /* full raw log text           */
    bool         lastLineWasSpin = false;  /* for \r overwrite handling   */

    /* ── live terminal dialog (all three set/cleared together) ─────────── */
    GtkWindow*     liveLogWindow = nullptr;  /* non-null while dialog is open */
    GtkTextBuffer* liveLogBuffer = nullptr;  /* text buffer of open dialog    */
    GtkTextView*   liveLogView   = nullptr;  /* text view for auto-scroll     */

    /* progress tracking (written by pipeline cb, read by animation timer) */
    std::atomic<double> targetFraction{0.0};
    std::atomic<double> displayFraction{0.0};
    std::atomic<int>    currentStep{0};
    std::atomic<int>    totalSteps{10};
    guint               animTimer = 0;      /* g_timeout id               */

    std::atomic_bool running{false};
    std::chrono::steady_clock::time_point runStart;
};

/* ── file-picker infrastructure ──────────────────────────────────────────── */

struct PickerCtx {
    GtkEntry* pathStore; /* hidden entry that stores the selected path         */
    GtkLabel* display;   /* visible label below the button showing chosen path */
    bool      isFolder;
    bool      csvOnly;
    std::string title;
};

static void picker_ctx_free(gpointer p) {
    delete static_cast<PickerCtx*>(p);
}

static void on_file_dialog_done(GObject* src, GAsyncResult* res, gpointer ud) {
    auto* ctx = static_cast<PickerCtx*>(ud);
    GtkFileDialog* dlg = GTK_FILE_DIALOG(src);
    GError* err = nullptr;
    GFile* file = ctx->isFolder
        ? gtk_file_dialog_select_folder_finish(dlg, res, &err)
        : gtk_file_dialog_open_finish(dlg, res, &err);

    if (file) {
        char* path = g_file_get_path(file);
        if (path) {
            gtk_editable_set_text(GTK_EDITABLE(ctx->pathStore), path);
            gtk_label_set_text(ctx->display, path);
            g_free(path);
        }
        g_object_unref(file);
    }
    if (err) g_error_free(err);
}

static void on_picker_button_clicked(GtkButton* btn, gpointer ud) {
    auto* ctx = static_cast<PickerCtx*>(ud);
    GtkWindow* win = GTK_WINDOW(gtk_widget_get_root(GTK_WIDGET(btn)));

    GtkFileDialog* dlg = gtk_file_dialog_new();
    gtk_file_dialog_set_title(dlg, ctx->title.c_str());

    if (!ctx->isFolder && ctx->csvOnly) {
        GtkFileFilter* f1 = gtk_file_filter_new();
        gtk_file_filter_set_name(f1, "CSV files (*.csv)");
        gtk_file_filter_add_pattern(f1, "*.csv");

        GtkFileFilter* f2 = gtk_file_filter_new();
        gtk_file_filter_set_name(f2, "All files");
        gtk_file_filter_add_pattern(f2, "*");

        GListStore* fs = g_list_store_new(GTK_TYPE_FILE_FILTER);
        g_list_store_append(fs, f1);
        g_list_store_append(fs, f2);
        gtk_file_dialog_set_filters(dlg, G_LIST_MODEL(fs));
        gtk_file_dialog_set_default_filter(dlg, f1);
        g_object_unref(fs);
        g_object_unref(f1);
        g_object_unref(f2);
    }

    if (ctx->isFolder)
        gtk_file_dialog_select_folder(dlg, win, nullptr, on_file_dialog_done, ctx);
    else
        gtk_file_dialog_open(dlg, win, nullptr, on_file_dialog_done, ctx);

    g_object_unref(dlg);
}

/*
 * create_file_picker
 * ─────────────────
 * Returns a GtkBox that looks like:
 *
 *   [  Choose Dataset (CSV)  ]   ← GtkButton, full width
 *   /home/user/data.csv          ← small path label updated on selection
 *
 * *outEntry is a hidden GtkEntry that stores the raw path string so that
 * build_args() can read it via gtk_editable_get_text() as usual.
 */
static GtkWidget* create_file_picker(const char* buttonLabel,
                                     GtkEntry**  outEntry,
                                     bool        isFolder = false,
                                     bool        csvOnly  = false) {
    /* hidden storage */
    GtkWidget* hidden = gtk_entry_new();
    gtk_widget_set_visible(hidden, false);
    *outEntry = GTK_ENTRY(hidden);

    /* path display */
    GtkWidget* pathLbl = gtk_label_new("No file selected.");
    gtk_widget_add_css_class(pathLbl, "path-display-label");
    gtk_label_set_ellipsize(GTK_LABEL(pathLbl), PANGO_ELLIPSIZE_MIDDLE);
    gtk_label_set_max_width_chars(GTK_LABEL(pathLbl), 80);
    gtk_widget_set_halign(pathLbl, GTK_ALIGN_START);

    /* button */
    GtkWidget* btn = gtk_button_new_with_label(buttonLabel);
    gtk_widget_add_css_class(btn, "picker-btn");
    gtk_widget_set_hexpand(btn, true);

    auto* ctx = new PickerCtx{GTK_ENTRY(hidden), GTK_LABEL(pathLbl),
                               isFolder, csvOnly, buttonLabel};
    g_object_set_data_full(G_OBJECT(btn), "picker-ctx", ctx, picker_ctx_free);
    g_signal_connect(btn, "clicked", G_CALLBACK(on_picker_button_clicked), ctx);

    /* assemble */
    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 4);
    gtk_box_append(GTK_BOX(vbox), btn);
    gtk_box_append(GTK_BOX(vbox), pathLbl);
    gtk_box_append(GTK_BOX(vbox), hidden);
    return vbox;
}

/* ── generic widget factories ─────────────────────────────────────────────── */

static GtkWidget* make_row(const char* labelText, GtkWidget* control) {
    GtkWidget* row = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 12);
    gtk_widget_add_css_class(row, "form-row");

    GtkWidget* lbl = gtk_label_new(labelText);
    gtk_label_set_xalign(GTK_LABEL(lbl), 0.0f);
    gtk_widget_set_size_request(lbl, 240, -1);
    gtk_widget_set_valign(lbl, GTK_ALIGN_CENTER);

    gtk_widget_set_hexpand(control, true);
    gtk_widget_set_valign(control, GTK_ALIGN_CENTER);

    gtk_box_append(GTK_BOX(row), lbl);
    gtk_box_append(GTK_BOX(row), control);
    return row;
}

static GtkWidget* make_section(const char* heading) {
    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_add_css_class(box, "section-box");
    if (heading && *heading) {
        GtkWidget* h = gtk_label_new(heading);
        gtk_widget_add_css_class(h, "section-heading");
        gtk_label_set_xalign(GTK_LABEL(h), 0.0f);
        gtk_widget_set_margin_bottom(h, 4);
        gtk_box_append(GTK_BOX(box), h);
    }
    return box;
}

static GtkWidget* make_tab_scroll(GtkWidget** inner) {
    GtkWidget* scroll = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                   GTK_POLICY_NEVER, GTK_POLICY_AUTOMATIC);
    GtkWidget* box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 16);
    gtk_widget_set_margin_start(box, 16); gtk_widget_set_margin_end(box, 16);
    gtk_widget_set_margin_top(box, 14);   gtk_widget_set_margin_bottom(box, 14);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scroll), box);
    if (inner) *inner = box;
    return scroll;
}

static GtkWidget* mk_entry(GtkEntry** out, const char* ph) {
    GtkWidget* w = gtk_entry_new();
    gtk_entry_set_placeholder_text(GTK_ENTRY(w), ph);
    *out = GTK_ENTRY(w); return w;
}

static GtkWidget* mk_switch(GtkSwitch** out, bool init) {
    GtkWidget* w = gtk_switch_new();
    gtk_switch_set_active(GTK_SWITCH(w), init);
    gtk_widget_set_halign(w, GTK_ALIGN_START);
    *out = GTK_SWITCH(w); return w;
}

static GtkWidget* mk_spin(GtkSpinButton** out,
                           double lo, double hi, double step,
                           double init, int digits = 0) {
    GtkAdjustment* adj = gtk_adjustment_new(init, lo, hi, step, step * 10, 0);
    GtkWidget* w = gtk_spin_button_new(adj, 1.0, digits);
    gtk_spin_button_set_value(GTK_SPIN_BUTTON(w), init);
    *out = GTK_SPIN_BUTTON(w); return w;
}

static GtkWidget* mk_drop(GtkDropDown** out,
                           std::initializer_list<const char*> items,
                           const char* initial) {
    GtkStringList* list = gtk_string_list_new(nullptr);
    guint sel = 0, i = 0;
    for (const char* v : items) {
        gtk_string_list_append(list, v);
        if (std::string(v) == initial) sel = i;
        ++i;
    }
    GtkWidget* w = gtk_drop_down_new(G_LIST_MODEL(list), nullptr);
    gtk_drop_down_set_selected(GTK_DROP_DOWN(w), sel);
    *out = GTK_DROP_DOWN(w); return w;
}

/* ── read helpers ──────────────────────────────────────────────────────────── */

static std::string drop_val(GtkDropDown* d) {
    GListModel* m = gtk_drop_down_get_model(d);
    guint sel = gtk_drop_down_get_selected(d);
    if (sel == GTK_INVALID_LIST_POSITION) return "";
    const char* s = gtk_string_list_get_string(GTK_STRING_LIST(m), sel);
    return s ? s : "";
}

static std::string text_buf(GtkTextView* v) {
    GtkTextBuffer* b = gtk_text_view_get_buffer(v);
    GtkTextIter s0, e0;
    gtk_text_buffer_get_bounds(b, &s0, &e0);
    char* t = gtk_text_buffer_get_text(b, &s0, &e0, FALSE);
    std::string out = t ? t : "";
    g_free(t);
    return out;
}

/* ── status helpers ──────────────────────────────────────────────────────── */

static void set_status(GuiState* st, const std::string& msg, bool ok) {
    gtk_label_set_text(st->statusLabel, msg.c_str());
    gtk_widget_remove_css_class(GTK_WIDGET(st->statusLabel), "status-idle");
    gtk_widget_remove_css_class(GTK_WIDGET(st->statusLabel), "status-ok");
    gtk_widget_remove_css_class(GTK_WIDGET(st->statusLabel), "status-err");
    gtk_widget_add_css_class(GTK_WIDGET(st->statusLabel),
                             ok ? "status-ok" : "status-err");
}

/* Show progress overlay / hide idle bar */
static void show_run_panel(GuiState* st, bool running) {
    gtk_widget_set_visible(st->idleBar,       !running);
    gtk_widget_set_visible(st->progressPanel,  running);
}

/* ── gui_handle_pipeline_line ─────────────────────────────────────────────
 * Called on the GTK main thread for every line emitted on stdout.
 * Spin lines (\r-terminated) carry the [Seldon] spinner progress.
 * All lines are accumulated in capturedLog for the terminal dialog.     */

/* Append or overwrite-last-line in a GtkTextBuffer — called on main thread. */
static void live_buf_append(GtkTextBuffer* buf, GtkTextView* tv,
                             const std::string& line, bool replaceLast) {
    if (replaceLast) {
        /* Delete from the start of the last line to the end of the buffer */
        GtkTextIter end;
        gtk_text_buffer_get_end_iter(buf, &end);
        GtkTextIter lineStart = end;
        gtk_text_iter_set_line_offset(&lineStart, 0);
        /* If lineStart == end the last line is empty — back up one line */
        if (gtk_text_iter_equal(&lineStart, &end) &&
            gtk_text_iter_backward_line(&lineStart)) {
            gtk_text_iter_set_line_offset(&lineStart, 0);
        }
        gtk_text_buffer_delete(buf, &lineStart, &end);
    }
    if (!line.empty()) {
        GtkTextIter end;
        gtk_text_buffer_get_end_iter(buf, &end);
        gtk_text_buffer_insert(buf, &end, (line + "\n").c_str(), -1);
    }
    /* Auto-scroll to bottom */
    GtkTextIter endAfter;
    gtk_text_buffer_get_end_iter(buf, &endAfter);
    GtkTextMark* ins = gtk_text_buffer_get_insert(buf);
    gtk_text_buffer_place_cursor(buf, &endAfter);
    gtk_text_view_scroll_to_mark(tv, ins, 0.0, FALSE, 0.0, 1.0);
}

static void gui_handle_pipeline_line(GuiState* st, const std::string& line,
                                      bool isSpin) {
    const bool replaceLast = isSpin && st->lastLineWasSpin;

    /* ── update capturedLog ── */
    {
        std::lock_guard<std::mutex> lk(st->logMutex);
        if (replaceLast) {
            auto n = st->capturedLog.rfind('\n', st->capturedLog.size() - 2);
            if (n != std::string::npos)
                st->capturedLog.resize(n + 1);
            else
                st->capturedLog.clear();
        }
        if (!line.empty()) {
            st->capturedLog += line;
            st->capturedLog += '\n';
        }
        st->lastLineWasSpin = isSpin;
    }

    /* ── push directly into the live terminal dialog if it is open ── */
    if (st->liveLogBuffer && st->liveLogView) {
        live_buf_append(st->liveLogBuffer, st->liveLogView, line, replaceLast);
    }
}

/* ── progress animation timer ─────────────────────────────────────────────
 * Fires every 30 ms while the pipeline is running.
 * Smoothly moves displayFraction toward targetFraction.                   */

static gboolean on_progress_tick(gpointer ud) {
    auto* st = static_cast<GuiState*>(ud);
    if (!st->running.load()) {
        st->animTimer = 0;
        return G_SOURCE_REMOVE;
    }

    double target  = st->targetFraction.load();
    double current = st->displayFraction.load();
    constexpr double speed = 0.008; /* fraction per tick (30ms) */
    if (current < target) {
        current = std::min(target, current + speed);
        st->displayFraction.store(current);
    }
    gtk_progress_bar_set_fraction(st->progressBar, current);
    return G_SOURCE_CONTINUE;
}

/* ── terminal log dialog ──────────────────────────────────────────────────── */

static void show_terminal_log(GuiState* st) {
    /* If the dialog is already open, just bring it to the front. */
    if (st->liveLogWindow) {
        gtk_window_present(st->liveLogWindow);
        return;
    }

    GtkWidget* dlg = gtk_window_new();
    gtk_window_set_title(GTK_WINDOW(dlg), "Terminal Output");
    gtk_window_set_transient_for(GTK_WINDOW(dlg), st->window);
    gtk_window_set_modal(GTK_WINDOW(dlg), false);
    gtk_window_set_default_size(GTK_WINDOW(dlg), 900, 560);

    /* header */
    GtkWidget* hdr = gtk_header_bar_new();
    gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(hdr), true);
    GtkWidget* hdrLbl = gtk_label_new("Terminal Output");
    gtk_widget_add_css_class(hdrLbl, "title");
    gtk_header_bar_set_title_widget(GTK_HEADER_BAR(hdr), hdrLbl);
    gtk_window_set_titlebar(GTK_WINDOW(dlg), hdr);

    /* log text view inside a scrolled window */
    GtkWidget* scroll = gtk_scrolled_window_new();
    gtk_widget_set_vexpand(scroll, true);
    gtk_widget_set_hexpand(scroll, true);
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(scroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);

    GtkWidget* tv = gtk_text_view_new();
    gtk_text_view_set_editable(GTK_TEXT_VIEW(tv), false);
    gtk_text_view_set_cursor_visible(GTK_TEXT_VIEW(tv), false);
    gtk_text_view_set_monospace(GTK_TEXT_VIEW(tv), true);
    gtk_widget_add_css_class(tv, "terminal-log-view");
    gtk_widget_set_margin_start(tv, 10);
    gtk_widget_set_margin_end(tv, 10);
    gtk_widget_set_margin_top(tv, 8);
    gtk_widget_set_margin_bottom(tv, 8);
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(scroll), tv);

    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_append(GTK_BOX(vbox), scroll);
    gtk_window_set_child(GTK_WINDOW(dlg), vbox);

    /* ── Register as the live sink BEFORE populating, to avoid races ── */
    GtkTextBuffer* buf = gtk_text_view_get_buffer(GTK_TEXT_VIEW(tv));
    st->liveLogBuffer = buf;
    st->liveLogView   = GTK_TEXT_VIEW(tv);
    st->liveLogWindow = GTK_WINDOW(dlg);

    /* Populate with everything captured so far, then register live updates */
    {
        std::lock_guard<std::mutex> lk(st->logMutex);
        gtk_text_buffer_set_text(buf, st->capturedLog.c_str(), -1);
    }
    /* Scroll to bottom of the pre-populated content */
    GtkTextIter end;
    gtk_text_buffer_get_end_iter(buf, &end);
    GtkTextMark* ins = gtk_text_buffer_get_insert(buf);
    gtk_text_buffer_place_cursor(buf, &end);
    gtk_text_view_scroll_to_mark(GTK_TEXT_VIEW(tv), ins, 0.0, FALSE, 0.0, 1.0);

    /* When the window is closed, unregister as the live sink */
    g_signal_connect(dlg, "destroy", G_CALLBACK(+[](GtkWidget*, gpointer ud) {
        auto* s = static_cast<GuiState*>(ud);
        s->liveLogBuffer = nullptr;
        s->liveLogView   = nullptr;
        s->liveLogWindow = nullptr;
    }), st);

    gtk_window_present(GTK_WINDOW(dlg));
}

/* ── build_args ───────────────────────────────────────────────────────────── */

static bool build_args(GuiState* st,
                        std::vector<std::string>& out,
                        std::string& err) {
    std::string dataset = trim(gtk_editable_get_text(GTK_EDITABLE(st->datasetEntry)));
    if (dataset.empty()) {
        err = "No dataset chosen — click \"Choose Dataset (CSV)\" on the Data tab.";
        return false;
    }

    out.clear();
    out.push_back("seldon");
    out.push_back(dataset);

    auto add = [&out](const std::string& k, const std::string& v) {
        if (!trim(v).empty()) { out.push_back(k); out.push_back(v); }
    };

    add("--target",      gtk_editable_get_text(GTK_EDITABLE(st->targetEntry)));
    add("--output-dir",  gtk_editable_get_text(GTK_EDITABLE(st->outputEntry)));
    add("--report",      gtk_editable_get_text(GTK_EDITABLE(st->reportEntry)));
    add("--assets-dir",  gtk_editable_get_text(GTK_EDITABLE(st->assetsEntry)));
    add("--delimiter",   gtk_editable_get_text(GTK_EDITABLE(st->delimiterEntry)));
    add("--exclude",     gtk_editable_get_text(GTK_EDITABLE(st->excludeEntry)));

    add("--profile",             drop_val(st->profileDrop));
    add("--export-preprocessed", drop_val(st->exportPreprocessedDrop));
    add("--target-strategy",     drop_val(st->targetStrategyDrop));
    add("--feature-strategy",    drop_val(st->featureStrategyDrop));
    add("--neural-strategy",     drop_val(st->neuralStrategyDrop));
    add("--bivariate-strategy",  drop_val(st->bivariateStrategyDrop));

    add("--fast",             bool_flag(gtk_switch_get_active(st->fastSwitch)));
    add("--low-memory",       bool_flag(gtk_switch_get_active(st->lowMemorySwitch)));
    add("--generate-html",    bool_flag(gtk_switch_get_active(st->htmlSwitch)));
    add("--verbose-analysis", bool_flag(gtk_switch_get_active(st->verboseSwitch)));
    add("--plot-univariate",  bool_flag(gtk_switch_get_active(st->plotUniSwitch)));
    add("--plot-overall",     bool_flag(gtk_switch_get_active(st->plotOverallSwitch)));
    add("--plot-bivariate",   bool_flag(gtk_switch_get_active(st->plotBiSwitch)));

    add("--plot-format", drop_val(st->plotFormatDrop));
    add("--plot-theme",  drop_val(st->plotThemeDrop));
    add("--plot-grid",   bool_flag(gtk_switch_get_active(st->plotGridSwitch)));
    add("--plot-width",  std::to_string(gtk_spin_button_get_value_as_int(st->plotWidthSpin)));
    add("--plot-height", std::to_string(gtk_spin_button_get_value_as_int(st->plotHeightSpin)));
    { std::ostringstream s; s << gtk_spin_button_get_value(st->plotPointSizeSpin);
      add("--plot-point-size", s.str()); }
    { std::ostringstream s; s << gtk_spin_button_get_value(st->plotLineWidthSpin);
      add("--plot-line-width", s.str()); }

    std::ostringstream yaml;
    yaml << "outlier_method: "       << drop_val(st->outlierMethodDrop)           << "\n"
         << "outlier_action: "       << drop_val(st->outlierActionDrop)           << "\n"
         << "scaling: "              << drop_val(st->scalingDrop)                  << "\n"
         << "kfold: "                << gtk_spin_button_get_value_as_int(st->kfoldSpin) << "\n"
         << "neural_optimizer: "     << drop_val(st->neuralOptimizerDrop)         << "\n"
         << "neural_ordinal_mode: "  << drop_val(st->neuralOrdinalModeDrop)       << "\n"
         << "neural_explainability: "<< drop_val(st->neuralExplainDrop)           << "\n"
         << "neural_learning_rate: " << gtk_spin_button_get_value(st->neuralLrSpin) << "\n"
         << "neural_min_layers: "    << gtk_spin_button_get_value_as_int(st->neuralMinLayersSpin) << "\n"
         << "neural_max_layers: "    << gtk_spin_button_get_value_as_int(st->neuralMaxLayersSpin) << "\n"
         << "neural_max_hidden_nodes: " << gtk_spin_button_get_value_as_int(st->neuralMaxHiddenSpin) << "\n"
         << "neural_streaming_mode: "   << bool_flag(gtk_switch_get_active(st->neuralStreamingSwitch)) << "\n"
         << "neural_multi_output: "     << bool_flag(gtk_switch_get_active(st->neuralMultiOutputSwitch)) << "\n"
         << "neural_ood_enabled: "      << bool_flag(gtk_switch_get_active(st->neuralOodSwitch)) << "\n"
         << "feature_engineering_enable_poly: "    << bool_flag(gtk_switch_get_active(st->featPolySwitch))  << "\n"
         << "feature_engineering_enable_log: "     << bool_flag(gtk_switch_get_active(st->featLogSwitch))   << "\n"
         << "feature_engineering_enable_ratio_product_discovery: "
              << bool_flag(gtk_switch_get_active(st->featRatioSwitch)) << "\n"
         << "feature_engineering_degree: "   << gtk_spin_button_get_value_as_int(st->featDegreeSpin)  << "\n"
         << "feature_engineering_max_base: " << gtk_spin_button_get_value_as_int(st->featMaxBaseSpin) << "\n";

    const std::string manualYaml = text_buf(st->overridesView);
    if (!trim(manualYaml).empty()) yaml << "\n" << manualYaml << "\n";

    std::string cfgPath = create_config_overlay(
        gtk_editable_get_text(GTK_EDITABLE(st->configEntry)),
        yaml.str(), err);
    if (!err.empty()) return false;
    if (!cfgPath.empty()) add("--config", cfgPath);

    for (auto& tok : split_args_inline(text_buf(st->extraArgsView)))
        out.push_back(tok);

    std::ostringstream prev;
    for (size_t i = 0; i < out.size(); ++i) {
        if (i) prev << ' ';
        bool q = out[i].find(' ') != std::string::npos;
        if (q) prev << '"'; prev << out[i]; if (q) prev << '"';
    }
    gtk_label_set_text(st->previewLabel, prev.str().c_str());
    return true;
}

/* ── run pipeline ─────────────────────────────────────────────────────────── */

struct RunFinished {
    GuiState*   state;
    bool        ok;
    std::string message;
    double      elapsedSeconds;
};

static gboolean on_run_finished(gpointer data) {
    std::unique_ptr<RunFinished> u(static_cast<RunFinished*>(data));
    auto* st = u->state;

    /* Uninstall the progress callback */
    AutomationPipeline::onProgress = nullptr;

    /* Stop animation timer */
    if (st->animTimer) {
        g_source_remove(st->animTimer);
        st->animTimer = 0;
    }

    /* Snap bar to full/zero */
    gtk_progress_bar_set_fraction(st->progressBar, u->ok ? 1.0 : 0.0);

    st->running.store(false);
    gtk_widget_set_sensitive(GTK_WIDGET(st->runButton), true);

    /* Build finish message */
    std::ostringstream msg;
    if (u->ok) {
        msg << "✓  Pipeline finished in "
            << std::fixed;
        msg.precision(1);
        msg << u->elapsedSeconds << " s";
    } else {
        msg << "✗  " << u->message;
    }
    set_status(st, msg.str(), u->ok);

    /* Switch back to idle panel */
    show_run_panel(st, false);

    /* Also add to captured log */
    {
        std::lock_guard<std::mutex> lk(st->logMutex);
        st->capturedLog += "\n--- " + msg.str() + " ---\n";
    }

    return G_SOURCE_REMOVE;
}

static void on_run_clicked(GtkButton*, gpointer ud) {
    auto* st = static_cast<GuiState*>(ud);
    if (st->running.load()) return;

    std::vector<std::string> args;
    std::string err;
    if (!build_args(st, args, err)) {
        set_status(st, err, false);
        return;
    }

    /* Reset run state */
    {
        std::lock_guard<std::mutex> lk(st->logMutex);
        st->capturedLog.clear();
        st->lastLineWasSpin = false;
    }
    st->targetFraction.store(0.0);
    st->displayFraction.store(0.0);
    st->currentStep.store(0);

    /* Update UI */
    st->running.store(true);
    gtk_widget_set_sensitive(GTK_WIDGET(st->runButton), false);
    gtk_label_set_text(st->stepLabel, "Initializing…");
    gtk_label_set_text(st->stepCountLabel, "Step 0 of 10");
    gtk_label_set_text(st->pctLabel, "0%");
    gtk_progress_bar_set_fraction(st->progressBar, 0.0);
    show_run_panel(st, true);

    /* Start smooth-animation timer (30 ms) */
    st->animTimer = g_timeout_add(30, on_progress_tick, st);

    /* Install progress callback */
    st->runStart = std::chrono::steady_clock::now();
    AutomationPipeline::onProgress =
        [st](const std::string& label, int step, int total) {
            /* Called on the pipeline thread — marshal to GTK main thread */
            struct ProgData { GuiState* st; std::string label; int step, total; };
            auto* d = new ProgData{st, label, step, total};
            g_main_context_invoke(nullptr, [](gpointer p) -> gboolean {
                std::unique_ptr<ProgData> pd(static_cast<ProgData*>(p));
                pd->st->currentStep.store(pd->step);
                pd->st->totalSteps.store(pd->total);
                double frac = pd->total > 0
                    ? static_cast<double>(pd->step) / static_cast<double>(pd->total)
                    : 0.0;
                pd->st->targetFraction.store(frac);

                /* Update step label */
                gtk_label_set_text(pd->st->stepLabel, pd->label.c_str());
                std::string cnt = "Step " + std::to_string(pd->step)
                                + " of " + std::to_string(pd->total);
                gtk_label_set_text(pd->st->stepCountLabel, cnt.c_str());
                int pct = static_cast<int>(frac * 100.0);
                gtk_label_set_text(pd->st->pctLabel,
                                   (std::to_string(pct) + "%").c_str());
                return G_SOURCE_REMOVE;
            }, d);
        };

    /* Install TeeStreambuf to capture stdout */
    std::streambuf* realBuf = std::cout.rdbuf();
    auto* tee = new TeeStreambuf(realBuf, st);
    std::cout.rdbuf(tee);

    std::thread([st, args = std::move(args), realBuf, tee]() mutable {
        auto* u = new RunFinished{st, false, "", 0.0};
        try {
            std::vector<char*> argv;
            argv.reserve(args.size());
            for (auto& t : args) argv.push_back(t.data());
            AutoConfig cfg = AutoConfig::fromArgs(static_cast<int>(argv.size()), argv.data());
            AutomationPipeline pipeline;
            int code = pipeline.run(cfg);
            u->ok      = (code == 0);
            u->message = u->ok ? "Run completed successfully."
                               : "Run failed (exit " + std::to_string(code) + ").";
        } catch (const Seldon::SeldonException& e) {
            u->ok = false; u->message = std::string("Config error: ") + e.what();
        } catch (const std::exception& e) {
            u->ok = false; u->message = std::string("Error: ") + e.what();
        } catch (...) {
            u->ok = false; u->message = "Unknown error.";
        }

        /* Restore stdout before calling tee's destructor */
        std::cout.flush();
        std::cout.rdbuf(realBuf);
        delete tee;

        auto now = std::chrono::steady_clock::now();
        u->elapsedSeconds = std::chrono::duration<double>(now - st->runStart).count();

        g_main_context_invoke(nullptr, on_run_finished, u);
    }).detach();
}

/* ── tab builders ─────────────────────────────────────────────────────────── */

static GtkWidget* build_data_tab(GuiState* st) {
    GtkWidget* page;
    GtkWidget* scroll = make_tab_scroll(&page);

    GtkWidget* s1 = make_section("Dataset");
    gtk_box_append(GTK_BOX(page), s1);
    /* Primary action — large, immediately visible */
    GtkWidget* dsPicker = create_file_picker(
        "Choose Dataset (CSV)", &st->datasetEntry, false, true);
    gtk_widget_add_css_class(
        gtk_widget_get_first_child(dsPicker), "primary-picker");
    gtk_box_append(GTK_BOX(s1), dsPicker);

    GtkWidget* s2 = make_section("Output");
    gtk_box_append(GTK_BOX(page), s2);
    gtk_box_append(GTK_BOX(s2),
        create_file_picker("Choose Output Folder", &st->outputEntry, true));

    GtkWidget* s3 = make_section("Columns & Formatting");
    gtk_box_append(GTK_BOX(page), s3);
    gtk_box_append(GTK_BOX(s3), make_row("Target column",
        mk_entry(&st->targetEntry, "e.g. price   (leave empty for unsupervised)")));
    gtk_box_append(GTK_BOX(s3), make_row("Exclude columns",
        mk_entry(&st->excludeEntry, "id,date_col   (comma-separated)")));
    gtk_box_append(GTK_BOX(s3), make_row("CSV delimiter",
        mk_entry(&st->delimiterEntry, ",")));
    gtk_box_append(GTK_BOX(s3), make_row("Report filename",
        mk_entry(&st->reportEntry, "report.md")));
    gtk_box_append(GTK_BOX(s3), make_row("Assets sub-directory",
        mk_entry(&st->assetsEntry, "seldon_report_assets")));
    gtk_box_append(GTK_BOX(s3), make_row("Export preprocessed",
        mk_drop(&st->exportPreprocessedDrop,
                {"none","csv","parquet"}, "none")));
    return scroll;
}

static GtkWidget* build_analysis_tab(GuiState* st) {
    GtkWidget* page;
    GtkWidget* scroll = make_tab_scroll(&page);

    GtkWidget* s1 = make_section("Execution");
    gtk_box_append(GTK_BOX(page), s1);
    gtk_box_append(GTK_BOX(s1), make_row("Profile",
        mk_drop(&st->profileDrop, {"auto","quick","thorough","minimal"}, "auto")));
    gtk_box_append(GTK_BOX(s1), make_row("Fast mode",
        mk_switch(&st->fastSwitch, false)));
    gtk_box_append(GTK_BOX(s1), make_row("Low memory",
        mk_switch(&st->lowMemorySwitch, false)));
    gtk_box_append(GTK_BOX(s1), make_row("Verbose stdout",
        mk_switch(&st->verboseSwitch, true)));
    gtk_box_append(GTK_BOX(s1), make_row("K-Fold chunks",
        mk_spin(&st->kfoldSpin, 2, 20, 1, 5)));

    GtkWidget* s2 = make_section("Strategy");
    gtk_box_append(GTK_BOX(page), s2);
    gtk_box_append(GTK_BOX(s2), make_row("Target strategy",
        mk_drop(&st->targetStrategyDrop,
                {"auto","quality","max_variance","last_numeric"}, "auto")));
    gtk_box_append(GTK_BOX(s2), make_row("Feature strategy",
        mk_drop(&st->featureStrategyDrop,
                {"auto","adaptive","aggressive","lenient"}, "auto")));
    gtk_box_append(GTK_BOX(s2), make_row("Bivariate strategy",
        mk_drop(&st->bivariateStrategyDrop,
                {"auto","balanced","corr_heavy","importance_heavy"}, "auto")));

    GtkWidget* s3 = make_section("Outliers & Scaling");
    gtk_box_append(GTK_BOX(page), s3);
    gtk_box_append(GTK_BOX(s3), make_row("Outlier method",
        mk_drop(&st->outlierMethodDrop,
                {"iqr","zscore","modified_zscore","adjusted_boxplot","lof"}, "iqr")));
    gtk_box_append(GTK_BOX(s3), make_row("Outlier action",
        mk_drop(&st->outlierActionDrop, {"flag","remove","cap"}, "flag")));
    gtk_box_append(GTK_BOX(s3), make_row("Scaling strategy",
        mk_drop(&st->scalingDrop,
                {"auto","zscore","minmax","none"}, "auto")));
    return scroll;
}

static GtkWidget* build_neural_tab(GuiState* st) {
    GtkWidget* page;
    GtkWidget* scroll = make_tab_scroll(&page);

    GtkWidget* s1 = make_section("Architecture");
    gtk_box_append(GTK_BOX(page), s1);
    gtk_box_append(GTK_BOX(s1), make_row("Strategy",
        mk_drop(&st->neuralStrategyDrop,
                {"auto","none","fast","balanced","expressive"}, "auto")));
    gtk_box_append(GTK_BOX(s1), make_row("Optimizer",
        mk_drop(&st->neuralOptimizerDrop, {"sgd","adam","lookahead"}, "lookahead")));
    gtk_box_append(GTK_BOX(s1), make_row("Learning rate",
        mk_spin(&st->neuralLrSpin, 0.00001, 1.0, 0.0001, 0.001, 5)));
    gtk_box_append(GTK_BOX(s1), make_row("Min hidden layers",
        mk_spin(&st->neuralMinLayersSpin, 1, 10, 1, 1)));
    gtk_box_append(GTK_BOX(s1), make_row("Max hidden layers",
        mk_spin(&st->neuralMaxLayersSpin, 1, 10, 1, 3)));
    gtk_box_append(GTK_BOX(s1), make_row("Max nodes per layer",
        mk_spin(&st->neuralMaxHiddenSpin, 4, 1024, 1, 128)));

    GtkWidget* s2 = make_section("Behaviour");
    gtk_box_append(GTK_BOX(page), s2);
    gtk_box_append(GTK_BOX(s2), make_row("Ordinal mode",
        mk_drop(&st->neuralOrdinalModeDrop,
                {"rank_regression","binary_cross_entropy_when_possible"},
                "rank_regression")));
    gtk_box_append(GTK_BOX(s2), make_row("Explainability",
        mk_drop(&st->neuralExplainDrop,
                {"permutation","integrated_gradients","hybrid"}, "hybrid")));
    gtk_box_append(GTK_BOX(s2), make_row("Batch streaming",
        mk_switch(&st->neuralStreamingSwitch, false)));
    gtk_box_append(GTK_BOX(s2), make_row("Multi-output targets",
        mk_switch(&st->neuralMultiOutputSwitch, true)));
    gtk_box_append(GTK_BOX(s2), make_row("OOD detection guard",
        mk_switch(&st->neuralOodSwitch, true)));
    return scroll;
}

static GtkWidget* build_features_tab(GuiState* st) {
    GtkWidget* page;
    GtkWidget* scroll = make_tab_scroll(&page);

    GtkWidget* s = make_section("Transformations");
    gtk_box_append(GTK_BOX(page), s);
    gtk_box_append(GTK_BOX(s), make_row("Polynomial discovery",
        mk_switch(&st->featPolySwitch,  true)));
    gtk_box_append(GTK_BOX(s), make_row("Log transformations",
        mk_switch(&st->featLogSwitch,   true)));
    gtk_box_append(GTK_BOX(s), make_row("Ratio & product pairs",
        mk_switch(&st->featRatioSwitch, true)));
    gtk_box_append(GTK_BOX(s), make_row("Polynomial degree",
        mk_spin(&st->featDegreeSpin,   1, 5,   1, 2)));
    gtk_box_append(GTK_BOX(s), make_row("Max base variants",
        mk_spin(&st->featMaxBaseSpin,  2, 100, 1, 8)));
    return scroll;
}

static GtkWidget* build_plots_tab(GuiState* st) {
    GtkWidget* page;
    GtkWidget* scroll = make_tab_scroll(&page);

    GtkWidget* s1 = make_section("Enable Charts");
    gtk_box_append(GTK_BOX(page), s1);
    gtk_box_append(GTK_BOX(s1), make_row("Univariate charts",
        mk_switch(&st->plotUniSwitch,     false)));
    gtk_box_append(GTK_BOX(s1), make_row("Bivariate charts",
        mk_switch(&st->plotBiSwitch,      true)));
    gtk_box_append(GTK_BOX(s1), make_row("Overall correlation",
        mk_switch(&st->plotOverallSwitch, false)));
    gtk_box_append(GTK_BOX(s1), make_row("HTML export",
        mk_switch(&st->htmlSwitch,        false)));

    GtkWidget* s2 = make_section("Rendering");
    gtk_box_append(GTK_BOX(page), s2);
    gtk_box_append(GTK_BOX(s2), make_row("Format",
        mk_drop(&st->plotFormatDrop, {"png","svg","pdf"}, "png")));
    gtk_box_append(GTK_BOX(s2), make_row("Theme",
        mk_drop(&st->plotThemeDrop,  {"auto","light","dark"}, "auto")));
    gtk_box_append(GTK_BOX(s2), make_row("Gridlines",
        mk_switch(&st->plotGridSwitch, true)));
    gtk_box_append(GTK_BOX(s2), make_row("Canvas width (px)",
        mk_spin(&st->plotWidthSpin,      320, 8192, 1, 1280)));
    gtk_box_append(GTK_BOX(s2), make_row("Canvas height (px)",
        mk_spin(&st->plotHeightSpin,     240, 8192, 1, 720)));
    gtk_box_append(GTK_BOX(s2), make_row("Marker size",
        mk_spin(&st->plotPointSizeSpin,  0.1, 8.0, 0.1, 0.8, 2)));
    gtk_box_append(GTK_BOX(s2), make_row("Line width",
        mk_spin(&st->plotLineWidthSpin,  0.1, 8.0, 0.1, 2.0, 2)));
    return scroll;
}

static GtkWidget* build_advanced_tab(GuiState* st) {
    GtkWidget* page;
    GtkWidget* scroll = make_tab_scroll(&page);

    GtkWidget* s1 = make_section("Base Config File");
    gtk_box_append(GTK_BOX(page), s1);
    gtk_box_append(GTK_BOX(s1),
        create_file_picker("Choose Config File (YAML / JSON)",
                           &st->configEntry));

    GtkWidget* s2 = make_section("Extra CLI Flags");
    gtk_box_append(GTK_BOX(page), s2);
    GtkWidget* argsScroll = gtk_scrolled_window_new();
    gtk_scrolled_window_set_policy(GTK_SCROLLED_WINDOW(argsScroll),
                                   GTK_POLICY_AUTOMATIC, GTK_POLICY_AUTOMATIC);
    gtk_widget_set_size_request(argsScroll, -1, 90);
    GtkWidget* argsView = gtk_text_view_new();
    st->extraArgsView = GTK_TEXT_VIEW(argsView);
    gtk_text_view_set_monospace(st->extraArgsView, true);
    gtk_text_view_set_wrap_mode(st->extraArgsView, GTK_WRAP_WORD_CHAR);
    gtk_widget_add_css_class(argsView, "code-view");
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(argsScroll), argsView);
    gtk_box_append(GTK_BOX(s2), argsScroll);

    GtkWidget* s3 = make_section("Direct YAML Overrides  (key: value, one per line)");
    gtk_box_append(GTK_BOX(page), s3);
    GtkWidget* ovScroll = gtk_scrolled_window_new();
    gtk_widget_set_size_request(ovScroll, -1, 130);
    gtk_widget_set_vexpand(ovScroll, true);
    GtkWidget* ovView = gtk_text_view_new();
    st->overridesView = GTK_TEXT_VIEW(ovView);
    gtk_text_view_set_monospace(st->overridesView, true);
    gtk_text_view_set_wrap_mode(st->overridesView, GTK_WRAP_WORD_CHAR);
    gtk_widget_add_css_class(ovView, "code-view");
    gtk_scrolled_window_set_child(GTK_SCROLLED_WINDOW(ovScroll), ovView);
    gtk_box_append(GTK_BOX(s3), ovScroll);
    return scroll;
}

/* ── activate ─────────────────────────────────────────────────────────────── */

static void activate(GtkApplication* app, gpointer ud) {
    auto* st = static_cast<GuiState*>(ud);
    st->app = app;

    /* Request Adwaita dark theme */
    g_object_set(gtk_settings_get_default(),
                 "gtk-application-prefer-dark-theme", TRUE,
                 nullptr);

    GtkWidget* win = gtk_application_window_new(app);
    st->window = GTK_WINDOW(win);
    gtk_window_set_title(st->window, "Seldon Analytics Studio");
    gtk_window_set_default_size(st->window, 1100, 780);

    /* ── header bar ─────────────────────────────────────────────────────── */
    GtkWidget* header = gtk_header_bar_new();
    gtk_header_bar_set_show_title_buttons(GTK_HEADER_BAR(header), true);

    GtkWidget* title_box = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    GtkWidget* title_lbl = gtk_label_new("Seldon Analytics");
    gtk_widget_add_css_class(title_lbl, "app-title");
    GtkWidget* sub_lbl = gtk_label_new("Data Intelligence & Modeling Engine");
    gtk_widget_add_css_class(sub_lbl, "app-subtitle");
    gtk_box_append(GTK_BOX(title_box), title_lbl);
    gtk_box_append(GTK_BOX(title_box), sub_lbl);
    gtk_header_bar_set_title_widget(GTK_HEADER_BAR(header), title_box);

    /* "Run Pipeline" lives in the header — always visible */
    st->runButton = GTK_BUTTON(gtk_button_new_with_label("  Run Pipeline  "));
    gtk_widget_add_css_class(GTK_WIDGET(st->runButton), "suggested-action");
    g_signal_connect(st->runButton, "clicked", G_CALLBACK(on_run_clicked), st);
    gtk_header_bar_pack_end(GTK_HEADER_BAR(header), GTK_WIDGET(st->runButton));

    gtk_window_set_titlebar(st->window, header);

    /* ── main layout ────────────────────────────────────────────────────── */
    GtkWidget* vbox = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);

    /* notebook */
    GtkWidget* nb = gtk_notebook_new();
    gtk_widget_set_vexpand(nb, true);
    gtk_notebook_append_page(GTK_NOTEBOOK(nb),
        build_data_tab(st),     gtk_label_new("Data"));
    gtk_notebook_append_page(GTK_NOTEBOOK(nb),
        build_analysis_tab(st), gtk_label_new("Analysis"));
    gtk_notebook_append_page(GTK_NOTEBOOK(nb),
        build_neural_tab(st),   gtk_label_new("Neural Network"));
    gtk_notebook_append_page(GTK_NOTEBOOK(nb),
        build_features_tab(st), gtk_label_new("Feature Engineering"));
    gtk_notebook_append_page(GTK_NOTEBOOK(nb),
        build_plots_tab(st),    gtk_label_new("Plots"));
    gtk_notebook_append_page(GTK_NOTEBOOK(nb),
        build_advanced_tab(st), gtk_label_new("Advanced"));
    gtk_box_append(GTK_BOX(vbox), nb);

    /* ── bottom container — holds idleBar and progressPanel, only one shown ─ */
    GtkWidget* bottomStack = gtk_box_new(GTK_ORIENTATION_VERTICAL, 0);
    gtk_widget_add_css_class(bottomStack, "bottom-stack");

    /* ── idleBar: command preview + status label ────────────────────────── */
    GtkWidget* idleBar = gtk_box_new(GTK_ORIENTATION_VERTICAL, 6);
    gtk_widget_add_css_class(idleBar, "idle-bar");
    gtk_widget_set_margin_start(idleBar, 14);  gtk_widget_set_margin_end(idleBar, 14);
    gtk_widget_set_margin_top(idleBar, 8);     gtk_widget_set_margin_bottom(idleBar, 10);
    st->idleBar = idleBar;

    /* preview row */
    GtkWidget* previewRow = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);
    GtkWidget* previewIcon = gtk_label_new("$");
    gtk_widget_add_css_class(previewIcon, "preview-icon");
    GtkWidget* prevLine = gtk_label_new("./seldon <dataset> [options…]");
    st->previewLabel = GTK_LABEL(prevLine);
    gtk_label_set_selectable(st->previewLabel, true);
    gtk_label_set_xalign(GTK_LABEL(prevLine), 0.0f);
    gtk_label_set_ellipsize(GTK_LABEL(prevLine), PANGO_ELLIPSIZE_END);
    gtk_widget_add_css_class(prevLine, "code-preview");
    gtk_widget_set_hexpand(prevLine, true);
    gtk_box_append(GTK_BOX(previewRow), previewIcon);
    gtk_box_append(GTK_BOX(previewRow), prevLine);
    gtk_box_append(GTK_BOX(idleBar), previewRow);

    /* status label */
    GtkWidget* statRow = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 6);
    GtkWidget* statusLbl = gtk_label_new("Ready");
    st->statusLabel = GTK_LABEL(statusLbl);
    gtk_widget_set_halign(statusLbl, GTK_ALIGN_START);
    gtk_widget_set_hexpand(statusLbl, true);
    gtk_widget_add_css_class(statusLbl, "status-idle");
    gtk_box_append(GTK_BOX(statRow), statusLbl);
    gtk_box_append(GTK_BOX(idleBar), statRow);

    gtk_box_append(GTK_BOX(bottomStack), idleBar);

    /* ── progressPanel: Ubuntu-installer style run panel ───────────────── */
    GtkWidget* progressPanel = gtk_box_new(GTK_ORIENTATION_VERTICAL, 10);
    gtk_widget_add_css_class(progressPanel, "run-panel");
    gtk_widget_set_margin_start(progressPanel, 14);  gtk_widget_set_margin_end(progressPanel, 14);
    gtk_widget_set_margin_top(progressPanel, 10);    gtk_widget_set_margin_bottom(progressPanel, 12);
    gtk_widget_set_visible(progressPanel, false);
    st->progressPanel = progressPanel;

    /* top row: title + terminal button */
    GtkWidget* topRow = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 8);

    GtkWidget* runTitle = gtk_label_new("Running Seldon Pipeline…");
    gtk_widget_add_css_class(runTitle, "run-title");
    gtk_widget_set_halign(runTitle, GTK_ALIGN_START);
    gtk_widget_set_hexpand(runTitle, true);

    GtkWidget* termBtn = gtk_button_new();
    gtk_widget_add_css_class(termBtn, "terminal-btn");
    gtk_widget_set_tooltip_text(termBtn, "Show full terminal output");
    /* terminal icon label */
    GtkWidget* termIcon = gtk_label_new(">_");
    gtk_widget_add_css_class(termIcon, "terminal-icon");
    gtk_button_set_child(GTK_BUTTON(termBtn), termIcon);
    st->terminalBtn = GTK_BUTTON(termBtn);
    g_signal_connect(termBtn, "clicked", G_CALLBACK(+[](GtkButton*, gpointer ud){
        show_terminal_log(static_cast<GuiState*>(ud));
    }), st);

    gtk_box_append(GTK_BOX(topRow), runTitle);
    gtk_box_append(GTK_BOX(topRow), termBtn);
    gtk_box_append(GTK_BOX(progressPanel), topRow);

    /* step label row */
    GtkWidget* stepRow = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);

    GtkWidget* stepBullet = gtk_label_new("●");
    gtk_widget_add_css_class(stepBullet, "step-bullet");

    GtkWidget* stepLbl = gtk_label_new("Initializing…");
    gtk_widget_add_css_class(stepLbl, "step-label");
    gtk_label_set_xalign(GTK_LABEL(stepLbl), 0.0f);
    gtk_widget_set_hexpand(stepLbl, true);
    st->stepLabel = GTK_LABEL(stepLbl);

    GtkWidget* stepCntLbl = gtk_label_new("Step 0 of 10");
    gtk_widget_add_css_class(stepCntLbl, "step-count");
    gtk_widget_set_halign(stepCntLbl, GTK_ALIGN_END);
    st->stepCountLabel = GTK_LABEL(stepCntLbl);

    gtk_box_append(GTK_BOX(stepRow), stepBullet);
    gtk_box_append(GTK_BOX(stepRow), stepLbl);
    gtk_box_append(GTK_BOX(stepRow), stepCntLbl);
    gtk_box_append(GTK_BOX(progressPanel), stepRow);

    /* progress bar row */
    GtkWidget* barRow = gtk_box_new(GTK_ORIENTATION_HORIZONTAL, 10);

    GtkWidget* bar = gtk_progress_bar_new();
    gtk_widget_add_css_class(bar, "run-progress-bar");
    gtk_widget_set_hexpand(bar, true);
    gtk_widget_set_valign(bar, GTK_ALIGN_CENTER);
    gtk_progress_bar_set_fraction(GTK_PROGRESS_BAR(bar), 0.0);
    st->progressBar = GTK_PROGRESS_BAR(bar);

    GtkWidget* pctLbl = gtk_label_new("0%");
    gtk_widget_add_css_class(pctLbl, "pct-label");
    gtk_widget_set_size_request(pctLbl, 40, -1);
    gtk_widget_set_halign(pctLbl, GTK_ALIGN_END);
    st->pctLabel = GTK_LABEL(pctLbl);

    gtk_box_append(GTK_BOX(barRow), bar);
    gtk_box_append(GTK_BOX(barRow), pctLbl);
    gtk_box_append(GTK_BOX(progressPanel), barRow);

    gtk_box_append(GTK_BOX(bottomStack), progressPanel);

    gtk_box_append(GTK_BOX(vbox), bottomStack);
    gtk_window_set_child(st->window, vbox);

    /* ── CSS ─────────────────────────────────────────────────────────────── */
    auto* css = gtk_css_provider_new();
    gtk_css_provider_load_from_string(css,

        /* ── header ── */
        ".app-title    { font-size: 15px; font-weight: 700; }"
        ".app-subtitle { font-size: 11px; opacity: 0.55; }"

        /* ── section heading ── */
        ".section-heading {"
        "  font-size: 10px; font-weight: 700;"
        "  text-transform: uppercase; letter-spacing: 1px; opacity: 0.50;"
        "}"

        /* ── section card ── */
        ".section-box {"
        "  border-radius: 8px;"
        "  border: 1px solid alpha(currentColor, 0.09);"
        "  padding: 10px 12px 12px 12px;"
        "}"

        /* ── form rows ── */
        ".form-row { padding: 5px 8px; border-radius: 5px; }"
        ".form-row > label { font-size: 14px; }"
        ".form-row:hover { background: alpha(currentColor, 0.04); }"

        /* ── picker buttons ── */
        ".picker-btn { min-height: 36px; font-size: 14px; border-radius: 8px; }"
        ".primary-picker { min-height: 48px; font-size: 15px; font-weight: 600; }"
        ".path-display-label {"
        "  font-family: monospace; font-size: 11px; opacity: 0.6; padding-left: 2px;"
        "}"

        /* ── bottom stack separator ── */
        ".bottom-stack { border-top: 1px solid alpha(currentColor, 0.09); }"

        /* ── idle bar ── */
        ".idle-bar { }"
        ".preview-icon {"
        "  font-family: monospace; font-size: 13px; opacity: 0.45;"
        "  padding-right: 2px;"
        "}"
        ".code-preview {"
        "  font-family: monospace; font-size: 13px; opacity: 0.70;"
        "}"
        ".status-idle   { font-size: 13px; opacity: 0.65; }"
        ".status-ok     { color: @success_color; font-weight: 600; font-size: 13px; }"
        ".status-err    { color: @error_color;   font-weight: 600; font-size: 13px; }"

        /* ── run panel ── */
        ".run-panel {"
        "  padding: 10px 14px 12px 14px;"
        "}"
        ".run-title { font-size: 13px; font-weight: 600; opacity: 0.80; }"

        /* terminal log button */
        ".terminal-btn {"
        "  border-radius: 6px; padding: 4px 10px;"
        "  border: 1px solid alpha(currentColor, 0.18);"
        "  background: alpha(currentColor, 0.06);"
        "  min-height: 28px;"
        "}"
        ".terminal-btn:hover  { background: alpha(currentColor, 0.12); }"
        ".terminal-icon { font-family: monospace; font-size: 12px; font-weight: 700; }"

        /* step row */
        ".step-bullet { color: @accent_color; font-size: 10px; opacity: 0.90; }"
        ".step-label  { font-size: 14px; font-weight: 500; }"
        ".step-count  { font-size: 12px; opacity: 0.55; }"

        /* progress bar */
        "progressbar.run-progress-bar {"
        "  min-height: 8px; border-radius: 4px;"
        "}"
        "progressbar.run-progress-bar > trough {"
        "  border-radius: 4px;"
        "  background: alpha(currentColor, 0.12);"
        "  min-height: 8px;"
        "}"
        "progressbar.run-progress-bar > trough > progress {"
        "  border-radius: 4px;"
        "  background: @accent_color;"
        "  min-height: 8px;"
        "}"
        ".pct-label { font-family: monospace; font-size: 13px; opacity: 0.70; }"

        /* code views (advanced tab) */
        ".code-view { font-family: monospace; font-size: 13px; }"

        /* terminal log dialog */
        ".terminal-log-view { font-family: monospace; font-size: 12.5px; }"
    );
    gtk_style_context_add_provider_for_display(
        gdk_display_get_default(),
        GTK_STYLE_PROVIDER(css),
        GTK_STYLE_PROVIDER_PRIORITY_APPLICATION);
    g_object_unref(css);

    gtk_window_present(st->window);
}

} // namespace

/* ── public entry point ───────────────────────────────────────────────────── */

int SeldonGui::run(int argc, char** argv) {
    auto st = std::make_unique<GuiState>();
    GtkApplication* app =
        gtk_application_new("com.seldon.studio", G_APPLICATION_DEFAULT_FLAGS);
    g_signal_connect(app, "activate", G_CALLBACK(activate), st.get());
    const int code = g_application_run(G_APPLICATION(app), argc, argv);
    g_object_unref(app);
    return code;
}

std::vector<std::string> SeldonGui::splitArgs(const std::string& text) {
    return split_args_inline(text);
}

std::string SeldonGui::toBool(bool value) { return bool_flag(value); }
