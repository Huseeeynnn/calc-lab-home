// Huseyn Jamilzade 251ADB038
// (optional) https://github.com/Huseeeynnn/calc-lab-home
// Compile with: gcc -O2 -Wall -Wextra -std=c17 -o calc calc.c -lm
// Resources used: // Resources used: AI (Claude, Anthropic): code review and rewriting of comments. Chat history: https://claude.ai/chat/f39d158b-e0a7-40d4-9d13-f5eada976d60

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <setjmp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define STUDENT_NAME "huseyn"
#define STUDENT_LASTNAME "jamilzade"
#define STUDENT_ID "251ADB038"

// Small utilities

static void die_oom(void) {
    fprintf(stderr, "calc: out of memory\n");
    exit(1);
}

static void *xrealloc(void *p, size_t n) {
    void *q = realloc(p, n);
    if (q == NULL) die_oom();
    return q;
}

// printf into a heap string, so paths need no fixed-size buffer.
static char *fmt(const char *f, ...) {
    va_list ap;
    size_t cap = 256;
    for (;;) {
        char *s = xrealloc(NULL, cap);
        va_start(ap, f);
        int n = vsnprintf(s, cap, f, ap);
        va_end(ap);
        if (n >= 0 && (size_t)n < cap) return s;
        free(s);
        if (n < 0) die_oom();
        cap = (size_t)n + 1;
    }
}

// Growable buffer: doubling capacity makes n appends cost O(n) in total,
// not O(n^2). This keeps multi-megabyte files fast.
typedef struct {
    char *data;
    size_t len, cap;
} Buf;

static void buf_append(Buf *b, const char *s, size_t n) {
    if (b->len + n > b->cap) {
        size_t cap = b->cap ? b->cap : 4096;
        while (cap < b->len + n) cap *= 2;
        b->data = xrealloc(b->data, cap);
        b->cap = cap;
    }
    memcpy(b->data + b->len, s, n);
    b->len += n;
}

// Values and arithmetic

// Int or double. The flag travels with the value, so each operation decides:
// both ints -> int arithmetic, otherwise both operands become double.
typedef struct {
    int is_float;
    long long i;
    double d;
} Value;

static double as_double(Value v) { return v.is_float ? v.d : (double)v.i; }

static Value make_int(long long i) {
    Value v = {0, i, 0.0};
    return v;
}

static Value make_float(double d) {
    Value v = {1, 0, d};
    return v;
}

// Error handling

// The first error ends the file, so fail() longjmps back to evaluate()
// instead of passing error codes up. Only the global g_stack is on the heap, so nothing leaks.
static jmp_buf g_jmp;
static size_t g_errpos;

static _Noreturn void fail(size_t pos) {
    g_errpos = pos;
    longjmp(g_jmp, 1);
}

// Tokenizer

typedef enum {
    T_INT, T_FLT, T_PLUS, T_MINUS, T_MUL, T_DIV, T_POW,
    T_LP, T_RP, T_END, T_BAD
} TokType;

typedef struct {
    TokType type;
    size_t pos;      // 1-based offset in the WHOLE file
    long long i;
    double d;
} Token;

// The file stays in one buffer and g_cur indexes it, so g_cur + 1 is the file
// position directly (newlines and comments included). Lines are never copied.
static char *g_src;
static size_t g_len, g_cur;
static Token tok;   // one-token lookahead, enough for this LL(1) grammar

// Never reports errors: unknown characters become T_BAD, and the parser fails
// only if it reaches one. So in "10 / 0 &" division by zero wins over '&'.
static void advance(void) {
    while (g_cur < g_len && (g_src[g_cur] == ' ' || g_src[g_cur] == '\t')) g_cur++;
    tok.pos = g_cur + 1;
    if (g_cur >= g_len) {
        // Last line without '\n': the position is file length + 1.
        tok.type = T_END;
        return;
    }
    unsigned char c = (unsigned char)g_src[g_cur];
    if (c == '\n') {
        tok.type = T_END;
        g_cur++;
        return;
    }
    if (isdigit(c)) {
        unsigned long long n = 0;
        size_t start = g_cur;
        while (g_cur < g_len && isdigit((unsigned char)g_src[g_cur])) {
            n = n * 10 + (unsigned long long)(g_src[g_cur] - '0');
            g_cur++;
        }
        // '.' starts a float only if a digit follows. "5." and "1.5e3" stop before
        // '.' / 'e', which then becomes T_BAD at the required position.
        if (g_cur + 1 < g_len && g_src[g_cur] == '.' &&
            isdigit((unsigned char)g_src[g_cur + 1])) {
            g_cur++;
            while (g_cur < g_len && isdigit((unsigned char)g_src[g_cur])) g_cur++;
            // Terminate in place for strtod instead of copying (read_file keeps a spare byte).
            char saved = g_src[g_cur];
            g_src[g_cur] = '\0';
            tok.d = strtod(g_src + start, NULL);
            g_src[g_cur] = saved;
            tok.type = T_FLT;
        } else {
            tok.i = (long long)n;
            tok.type = T_INT;
        }
        return;
    }
    g_cur++;
    switch (c) {
    case '+': tok.type = T_PLUS; break;
    case '-': tok.type = T_MINUS; break;
    case '/': tok.type = T_DIV; break;
    case '(': tok.type = T_LP; break;
    case ')': tok.type = T_RP; break;
    case '*':
        // Longest match: "**" is one token, otherwise a single '*'.
        if (g_cur < g_len && g_src[g_cur] == '*') {
            g_cur++;
            tok.type = T_POW;
        } else {
            tok.type = T_MUL;
        }
        break;
    default: tok.type = T_BAD; break;
    }
}

// Operations

// Integer arithmetic is done in unsigned long long and converted back, so a
// wrap-around can never be undefined behaviour (the sanitizer build stays clean).
static Value apply_add_sub_mul(TokType op, Value l, Value r, size_t oppos) {
    if (!l.is_float && !r.is_float) {
        unsigned long long a = (unsigned long long)l.i, b = (unsigned long long)r.i, x;
        x = op == T_PLUS ? a + b : op == T_MINUS ? a - b : a * b;
        return make_int((long long)x);
    }
    double a = as_double(l), b = as_double(r);
    double x = op == T_PLUS ? a + b : op == T_MINUS ? a - b : a * b;
    // Overflow to inf (or NaN) is reported at the operator, as the rules say.
    if (!isfinite(x)) fail(oppos);
    return make_float(x);
}

// rpos is the start of the right operand: that is where division by zero is
// reported, not at the '/' itself.
static Value apply_div(Value l, Value r, size_t oppos, size_t rpos) {
    if (!l.is_float && !r.is_float) {
        if (r.i == 0) fail(rpos);
        // LLONG_MIN / -1 would overflow (undefined in C); negate via unsigned instead.
        if (r.i == -1) return make_int((long long)(0ULL - (unsigned long long)l.i));
        return make_int(l.i / r.i);   // C truncates toward zero, as required
    }
    double b = as_double(r);
    if (b == 0.0) fail(rpos);
    double x = as_double(l) / b;
    if (!isfinite(x)) fail(oppos);
    return make_float(x);
}

// rstart is where the exponent begins (negative integer exponent is reported there),
// oppos is the '**' (non-finite float result is reported there).
static Value apply_pow(Value l, Value r, size_t oppos, size_t rstart) {
    if (!l.is_float && !r.is_float) {
        if (r.i < 0) fail(rstart);
        // Square-and-multiply: O(log exponent), so something like 1 ** 10000000000 is instant.
        unsigned long long result = 1, b = (unsigned long long)l.i;
        unsigned long long e = (unsigned long long)r.i;
        while (e) {
            if (e & 1) result *= b;
            b *= b;
            e >>= 1;
        }
        return make_int((long long)result);   // x ** 0 == 1, also 0 ** 0
    }
    double x = pow(as_double(l), as_double(r));
    if (!isfinite(x)) fail(oppos);
    return make_float(x);
}

// Parser (recursive descent, one function per precedence level)
//   expr    := term  { ('+' | '-') term }
//   term    := power { ('*' | '/') power }
//   power   := primary { '**' primary }   (equivalent to primary [ '**' power ])
//   primary := NUMBER | '(' expr ')'
// Lower precedence calls higher, so '*' and '**' are already evaluated when
// '+' is applied. There is no unary minus: '-' fails in primary().

static Value expr(void);

static Value primary(void) {
    if (tok.type == T_INT) {
        Value v = make_int(tok.i);
        advance();
        return v;
    }
    if (tok.type == T_FLT) {
        Value v = make_float(tok.d);
        advance();
        return v;
    }
    if (tok.type == T_LP) {
        advance();
        Value v = expr();
        // Missing ')' is reported at whatever token stands where ')' should be:
        // the line's '\n' (or file length + 1), or a bad character.
        if (tok.type != T_RP) fail(tok.pos);
        advance();
        return v;
    }
    // Missing operand, two operators in a row, "()", unary minus, invalid
    // character: in every case the current token is the one that cannot start an operand.
    fail(tok.pos);
}

// A '**' chain is collected on a heap stack and folded right to left, so
// 2 ** 3 ** 2 == 2 ** (3 ** 2). Recursing per '**' could overflow the C stack
// on very long chains.
typedef struct {
    Value v;
    size_t start;   // file position where this operand begins
    size_t oppos;   // file position of the '**' in front of it (unused for the first)
} PowItem;

static PowItem *g_stack;
static size_t g_sp, g_cap;

static void stack_push(Value v, size_t start, size_t oppos) {
    if (g_sp == g_cap) {
        g_cap = g_cap ? g_cap * 2 : 64;
        g_stack = xrealloc(g_stack, g_cap * sizeof *g_stack);
    }
    g_stack[g_sp].v = v;
    g_stack[g_sp].start = start;
    g_stack[g_sp].oppos = oppos;
    g_sp++;
}

static Value power(void) {
    size_t start = tok.pos;
    Value first = primary();
    if (tok.type != T_POW) return first;   // common case: no stack traffic

    // Nested parentheses call power() again, so each call remembers its own base.
    size_t base = g_sp;
    stack_push(first, start, 0);
    while (tok.type == T_POW) {
        size_t oppos = tok.pos;
        advance();
        size_t s = tok.pos;
        Value v = primary();
        stack_push(v, s, oppos);
    }
    // Fold right to left; indexes, not pointers, because the stack may be
    // reallocated by nested calls while we were collecting.
    Value r = g_stack[g_sp - 1].v;
    for (size_t i = g_sp - 1; i > base; i--)
        r = apply_pow(g_stack[i - 1].v, r, g_stack[i].oppos, g_stack[i].start);
    g_sp = base;
    return r;
}

static Value term(void) {
    Value l = power();
    while (tok.type == T_MUL || tok.type == T_DIV) {
        TokType op = tok.type;
        size_t oppos = tok.pos;
        advance();
        size_t rpos = tok.pos;
        Value r = power();
        // Evaluating immediately (left to right) gives left associativity.
        l = op == T_MUL ? apply_add_sub_mul(op, l, r, oppos) : apply_div(l, r, oppos, rpos);
    }
    return l;
}

static Value expr(void) {
    Value l = term();
    while (tok.type == T_PLUS || tok.type == T_MINUS) {
        TokType op = tok.type;
        size_t oppos = tok.pos;
        advance();
        Value r = term();
        l = apply_add_sub_mul(op, l, r, oppos);
    }
    return l;
}

static Buf *g_out;

static void emit(Value v) {
    char tmp[64];
    int n = v.is_float ? snprintf(tmp, sizeof tmp, "%.15g\n", v.d)
                       : snprintf(tmp, sizeof tmp, "%lld\n", v.i);
    buf_append(g_out, tmp, (size_t)n);
}

static void run_lines(void) {
    while (g_cur < g_len) {
        while (g_cur < g_len && (g_src[g_cur] == ' ' || g_src[g_cur] == '\t')) g_cur++;
        if (g_cur >= g_len) break;
        char c = g_src[g_cur];
        if (c == '\n') {   // blank line: no output
            g_cur++;
            continue;
        }
        if (c == '#') {    // comment only if '#' is the first non-blank character
            while (g_cur < g_len && g_src[g_cur] != '\n') g_cur++;
            if (g_cur < g_len) g_cur++;
            continue;
        }
        advance();
        Value v = expr();
        // Anything left on the line (extra ')', a second number, a bad
        // character, a '#' after an expression) is an error at that token.
        if (tok.type != T_END) fail(tok.pos);
        emit(v);
    }
}

// Results are buffered so an error can replace all output with one ERROR line.
// Parser state is global, so no locals change between setjmp and longjmp.
static void evaluate(char *text, size_t len, Buf *out) {
    g_src = text;
    g_len = len;
    g_cur = 0;
    g_sp = 0;
    g_out = out;
    if (setjmp(g_jmp) == 0) {
        run_lines();
        return;
    }
    out->len = 0;
    char tmp[64];
    int n = snprintf(tmp, sizeof tmp, "ERROR:%zu\n", g_errpos);
    buf_append(out, tmp, (size_t)n);
}

// Files and command line

// Reads the whole file with a doubling buffer (no size guess needed, no
// repeated copying). One extra byte is kept for the in-place terminator
// used by the float tokenizer.
static char *read_file(const char *path, size_t *len_out) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "calc: cannot open '%s': %s\n", path, strerror(errno));
        return NULL;
    }
    size_t cap = 1 << 16, len = 0;
    char *data = xrealloc(NULL, cap);
    for (;;) {
        if (len + 1 >= cap) {
            cap *= 2;
            data = xrealloc(data, cap);
        }
        size_t got = fread(data + len, 1, cap - len - 1, f);
        len += got;
        if (got == 0) break;
    }
    if (ferror(f)) {
        fprintf(stderr, "calc: cannot read '%s': %s\n", path, strerror(errno));
        fclose(f);
        free(data);
        return NULL;
    }
    fclose(f);
    data[len] = '\0';
    *len_out = len;
    return data;
}

// Last path component; trailing slashes are ignored, ".txt" optionally removed.
static char *base_name(const char *path, int strip_txt) {
    size_t n = strlen(path);
    while (n > 1 && path[n - 1] == '/') n--;
    size_t start = n;
    while (start > 0 && path[start - 1] != '/') start--;
    size_t len = n - start;
    if (strip_txt && len > 4 && strncmp(path + start + len - 4, ".txt", 4) == 0) len -= 4;
    return fmt("%.*s", (int)len, path + start);
}

static int ensure_dir(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        if (S_ISDIR(st.st_mode)) return 0;
        fprintf(stderr, "calc: '%s' exists and is not a folder\n", path);
        return -1;
    }
    if (mkdir(path, 0777) != 0) {
        fprintf(stderr, "calc: cannot create folder '%s': %s\n", path, strerror(errno));
        return -1;
    }
    return 0;
}

static int process_file(const char *inpath, const char *outdir) {
    size_t len;
    char *text = read_file(inpath, &len);
    if (text == NULL) return -1;

    Buf out = {NULL, 0, 0};
    evaluate(text, len, &out);
    free(text);

    int rc = -1;
    char *name = base_name(inpath, 1);
    char *outpath = fmt("%s/%s_%s_%s_%s.txt", outdir, name, STUDENT_NAME, STUDENT_LASTNAME, STUDENT_ID);
    if (ensure_dir(outdir) == 0) {
        FILE *f = fopen(outpath, "wb");
        if (f == NULL) {
            fprintf(stderr, "calc: cannot write '%s': %s\n", outpath, strerror(errno));
        } else {
            size_t w = out.len ? fwrite(out.data, 1, out.len, f) : 0;
            if (fclose(f) != 0 || w != out.len)
                fprintf(stderr, "calc: error writing '%s'\n", outpath);
            else
                rc = 0;
        }
    }
    free(outpath);
    free(name);
    free(out.data);
    return rc;
}

static int has_txt_suffix(const char *s) {
    size_t n = strlen(s);
    return n > 4 && strcmp(s + n - 4, ".txt") == 0;
}

static int process_dir(const char *dir, const char *outdir) {
    DIR *d = opendir(dir);
    if (d == NULL) {
        fprintf(stderr, "calc: cannot open folder '%s': %s\n", dir, strerror(errno));
        return -1;
    }
    if (ensure_dir(outdir) != 0) {
        closedir(d);
        return -1;
    }
    int rc = 0;
    struct dirent *e;
    while (rc == 0 && (e = readdir(d)) != NULL) {
        if (!has_txt_suffix(e->d_name)) continue;
        char *path = fmt("%s/%s", dir, e->d_name);
        struct stat st;
        // Only regular files directly inside DIR; a folder called x.txt is skipped.
        if (stat(path, &st) == 0 && S_ISREG(st.st_mode)) rc = process_file(path, outdir);
        free(path);
    }
    closedir(d);
    return rc;
}

static int usage(const char *msg) {
    if (msg) fprintf(stderr, "calc: %s\n", msg);
    fprintf(stderr, "usage: calc [-d DIR | --dir DIR] [-o OUTDIR | --output-dir OUTDIR] [input.txt]\n");
    return 1;
}

int main(int argc, char **argv) {
    const char *dir = NULL, *outdir = NULL, *input = NULL;
    if (argc < 2) return usage("no arguments");
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (strcmp(a, "-d") == 0 || strcmp(a, "--dir") == 0) {
            if (i + 1 >= argc) return usage("missing value for -d");
            dir = argv[++i];
        } else if (strcmp(a, "-o") == 0 || strcmp(a, "--output-dir") == 0) {
            if (i + 1 >= argc) return usage("missing value for -o");
            outdir = argv[++i];
        } else if (a[0] == '-' && a[1] != '\0') {
            fprintf(stderr, "calc: unknown option '%s'\n", a);
            return usage(NULL);
        } else if (input == NULL) {
            input = a;
        } else {
            return usage("more than one input file");
        }
    }
    if (dir == NULL && input == NULL) return usage("no input file or folder");
    if (dir != NULL && input != NULL) return usage("use either -d DIR or an input file, not both");

    char *default_out = NULL;
    if (outdir == NULL) {
        char *b = base_name(dir ? dir : input, dir == NULL);
        default_out = fmt("%s_%s_%s", b, STUDENT_NAME, STUDENT_ID);
        free(b);
        outdir = default_out;
    }
    int rc = dir ? process_dir(dir, outdir) : process_file(input, outdir);
    free(default_out);
    free(g_stack);
    return rc == 0 ? 0 : 1;
}