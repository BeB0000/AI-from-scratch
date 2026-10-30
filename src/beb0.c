// --------------------
// Copyright @BeB0000
// UNDER MIT LICENSE
// beb0.c
// --------------------
#define _POSIX_C_SOURCE 200809L
#define _BSD_SOURCE
#define _DEFAULT_SOURCE

#include "../include/beb0.h"

// ==globals==
// set to 1 by the signal handler so the loops know when to stop
static volatile int g_stop = 0;

// ==signals==
static void handle_signal(int s)
{
	(void)s;
	g_stop = 1;
}

// ==thermal protection==
// read the cpu temp from sysfs, returns -1 if it cant be read
static int cpu_temperature_c(void)
{
	FILE *f = fopen(THERMAL_PATH, "r");
	if (!f) return -1;
	int t = 0;
	fscanf(f, "%d", &t);
	fclose(f);
	// sysfs gives millidegrees
	return t / 1000;
}

// slow down (and lower the lr) if the cpu is getting too hot
static void thermal_check(float *learning_rate)
{
	int t = cpu_temperature_c();
	if (t < 0) return;
	// critical, sleep long and halve the lr
	if (t >= THERMAL_CRIT_C)
	{
		fprintf(stderr, "\n[thermal] %dc critical sleeping 750ms cpu is melting better slow down...\n", t);
		usleep(THERMAL_SLEEP_US * 5);
		*learning_rate = fmaxf(*learning_rate * 0.5f, FIRE_LEARNING_RATE);
	} else if (t >= THERMAL_WARN_C)
	{
		// warning, short nap
		usleep(THERMAL_SLEEP_US);
	}
}

// ==tokenizer==
#define BPE_PIECE_MAX 64

typedef struct
{
	char piece[BPE_PIECE_MAX];
	float score;
	int id;
	int special;
} BpeToken;

typedef struct
{
	BpeToken *tokens;
	int size;
} BpeVocab;

// load the vocab file, one "piece score" per line, line number = token id
static BpeVocab *vocab_load(const char *path)
{
	// clear the screen
	printf("\033[H\033[2J");
	FILE *f = fopen(path, "r");
	if (!f)
	{
		fprintf(stderr, "cannot open vocab %s did you forget to create it\n", path);
		return NULL;
	}
	BpeVocab *vocab = malloc(sizeof(BpeVocab));
	vocab->tokens = malloc(VOCAB_SIZE * sizeof(BpeToken));
	vocab->size = 0;
	char line[256];
	// == read every line ==
	while (fgets(line, sizeof(line), f) && vocab->size < VOCAB_SIZE)
	{
		BpeToken *tok = &vocab->tokens[vocab->size];
		float sc = 0.0f;

		// the score is optional
		if (sscanf(line, "%63s %f", tok->piece, &sc) >= 1)
		{
			tok->score = sc;
			tok->id = vocab->size;
			int len = strlen(tok->piece);
			// tokens like <eos> are special
			tok->special = (tok->piece[0] == '<' && tok->piece[len - 1] == '>');
			vocab->size++;
		}
	}
	fclose(f);
	printf("vocab has %d weird tokens yay\n", vocab->size);
	return vocab;
}

static void vocab_free(BpeVocab *vocab)
{
	free(vocab->tokens);
	free(vocab);
}

// linear search for a piece, returns TOKEN_UNK if not found
static int vocab_find(BpeVocab *vocab, const char *piece)
{
	for (int i = 0; i < vocab->size; i++)
	{
		if (strcmp(vocab->tokens[i].piece, piece) == 0) return i;
	}
	return TOKEN_UNK;
}

// greedy longest match encoder, tries the plain piece then the sentencepiece one
static int vocab_encode(BpeVocab *vocab, const char *text, int *out, int max_out)
{
	int len = strlen(text);
	int n = 0;
	int i = 0;
	while (i < len && n < max_out)
	{
		char piece[BPE_PIECE_MAX];
		int best_len = 0;
		int best_id = TOKEN_UNK;
		int try_len = (len - i < BPE_PIECE_MAX - 1) ? len - i : BPE_PIECE_MAX - 1;

		// == try the longest piece first, then shrink ==
		for (int l = try_len; l >= 1; l--)
		{
			strncpy(piece, text + i, l);
			piece[l] = '\0';
			int id = vocab_find(vocab, piece);
			if (id != TOKEN_UNK)
			{
				best_len = l;
				best_id = id;
				break;
			}

			// same piece but with the word start marker in front
			char sp[BPE_PIECE_MAX];
			sp[0] = '\xe2';
			sp[1] = '\x96';
			sp[2] = '\x81';
			strncpy(sp + 3, text + i, l);
			sp[l + 3] = '\0';
			id = vocab_find(vocab, sp);
			if (id != TOKEN_UNK)
			{
				best_len = l;
				best_id = id;
				break;
			}
		}
		out[n++] = best_id;
		// always move forward at least 1 char
		i += best_len > 0 ? best_len : 1;
	}
	return n;
}

// ==corpus==
typedef struct
{
	int32_t *tokens;
	long count;
	int fd;
	size_t mapped_len;
	int is_mmap;
} Corpus;

// load the token file with mmap, falls back to reading it into ram
static Corpus corpus_load(const char *path)
{
	Corpus c = {NULL, 0, -1, 0, 0};
	int fd = open(path, O_RDONLY);
	if (fd < 0)
	{
		fprintf(stderr, "cannot open corpus %s maybe its hiding\n", path);
		return c;
	}
	struct stat st;
	fstat(fd, &st);
	size_t sz = st.st_size;
	void *p = mmap(NULL, sz, PROT_READ, MAP_PRIVATE, fd, 0);
	// mmap failed, read the whole file instead
	if (p == MAP_FAILED)
	{
		FILE *f = fdopen(fd, "rb");
		c.tokens = malloc(sz);
		fread(c.tokens, 1, sz, f);
		fclose(f);
		c.count = sz / sizeof(int32_t);
		return c;
	}
	// we jump around randomly so tell the kernel
	madvise(p, sz, MADV_RANDOM);
	c.tokens = p;
	c.count = sz / sizeof(int32_t);
	c.fd = fd;
	c.mapped_len = sz;
	c.is_mmap = 1;
	printf("corpus %ld tokens thats a lot of words %.1f mb\n", c.count, (double)sz / 1e6);
	return c;
}

static void corpus_free(Corpus *c)
{
	if (!c->tokens) return;
	if (c->is_mmap)
	{
		munmap(c->tokens, c->mapped_len);
		close(c->fd);
	} else
	{
		free(c->tokens);
	}
	c->tokens = NULL;
}

// fill input/target with B random windows of T tokens, target is input shifted by 1
static void corpus_batch(Corpus *c, int *input, int *target, int B, int T)
{
	for (int b = 0; b < B;)
	{
		long start = (long)rand() % (c->count - T - 1);
		int bad = 0;
		for (int t = 0; t < T; t++)
		{
			int tok = (int)c->tokens[start + t];
			// skip windows with pad or out of range tokens
			if (tok == TOKEN_PAD || tok < 0 || tok >= VOCAB_SIZE)
			{
				bad = 1;
				break;
			}
			input[b * T + t] = tok;
			target[b * T + t] = (int)c->tokens[start + t + 1];
		}
		if (!bad) b++;
	}
}

// ==parameters==
// weights, gradients and the 2 adam moments
typedef struct
{
	float *w;
	float *g;
	float *m;
	float *v;
	int size;
} Parameter;

static void parameter_alloc(Parameter *p, int size)
{
	p->size = size;
	p->w = calloc(size, sizeof(float));
	p->g = calloc(size, sizeof(float));
	p->m = calloc(size, sizeof(float));
	p->v = calloc(size, sizeof(float));
	if (!p->w || !p->g || !p->m || !p->v)
	{
		fprintf(stderr, "out of memory i need %d floats but the hamster died\n", size);
		exit(1);
	}
}

static void parameter_free(Parameter *p)
{
	free(p->w);
	free(p->g);
	free(p->m);
	free(p->v);
}

// kaiming init using the box muller (polar) gaussian
static void parameter_kaiming(Parameter *p, int fan_in)
{
	float scale = sqrtf(2.0f / (float)fan_in);
	// box muller makes 2 numbers at a time, keep the spare one
	static int toggle = 0;
	static double spare;
	for (int i = 0; i < p->size; i++)
	{
		// use the spare from last time
		if (toggle)
		{
			p->w[i] = (float)(spare * scale);
			toggle = 0;
			continue;
		}
		double u, v, s;
		// pick a point inside the unit circle
		do
		{
			u = (rand() + 1.0) / (RAND_MAX + 2.0) * 2.0 - 1.0;
			v = (rand() + 1.0) / (RAND_MAX + 2.0) * 2.0 - 1.0;
			s = u * u + v * v;
		} while (s >= 1.0 || s == 0.0);
		s = sqrt(-2.0 * log(s) / s);
		p->w[i] = (float)(u * s * scale);
		spare = v * s;
		toggle = 1;
	}
}

// set every weight to 1 (used for layer norm scale)
static void parameter_ones(Parameter *p)
{
	for (int i = 0; i < p->size; i++) p->w[i] = 1.0f;
}

// ==math helpers==
static void softmax_inplace(float *x, int n)
{
	// subtract the max so exp doesnt explode
	float mx = x[0];
	for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
	float sum = 0.0f;
	for (int i = 0; i < n; i++)
	{
		x[i] = expf(x[i] - mx);
		sum += x[i];
	}
	float inv = 1.0f / sum;
	for (int i = 0; i < n; i++) x[i] *= inv;
}

// ==layer norm==
static void layer_norm_forward(float *out, float *mean_out, float *rstd_out,
const float *x, const float *scale, const float *bias, int n)
{
	// mean
	float mean = 0.0f;
	for (int i = 0; i < n; i++) mean += x[i];
	mean /= (float)n;
	// variance
	float var = 0.0f;
	for (int i = 0; i < n; i++)
	{
		float d = x[i] - mean;
		var += d * d;
	}
	var /= (float)n;
	float rstd = 1.0f / sqrtf(var + LN_EPS);
	// save mean and rstd for the backward pass
	*mean_out = mean;
	*rstd_out = rstd;
	// normalize then scale and shift
	for (int i = 0; i < n; i++)
	{
		out[i] = (x[i] - mean) * rstd * scale[i] + bias[i];
	}
}

static void layer_norm_backward(float *dx, float *dscale, float *dbias,
const float *dout, const float *x,
const float *scale, float mean, float rstd, int n)
{
	float sum_dxhat = 0.0f;
	float sum_dxhat_xhat = 0.0f;
	float xhat[HIDDEN_DIM];
	float dxhat[HIDDEN_DIM];
	// == first pass, grads for scale and bias + the sums ==
	for (int i = 0; i < n; i++)
	{
		xhat[i] = (x[i] - mean) * rstd;
		dxhat[i] = dout[i] * scale[i];
		if (dscale) dscale[i] += dout[i] * xhat[i];
		if (dbias) dbias[i] += dout[i];
		sum_dxhat += dxhat[i];
		sum_dxhat_xhat += dxhat[i] * xhat[i];
	}
	// == second pass, grad for the input ==
	float inv_n = 1.0f / (float)n;
	for (int i = 0; i < n; i++)
	{
		dx[i] += rstd * (dxhat[i] - inv_n * (sum_dxhat + xhat[i] * sum_dxhat_xhat));
	}
}

// ==linear layer==
// out = in * w + b, w is stored as [in_dim][out_dim]
static void matmul_forward(float *out, const float *in, const float *w, const float *b,
int in_dim, int out_dim)
{
	for (int o = 0; o < out_dim; o++)
	{
		// bias is optional
		float acc = b ? b[o] : 0.0f;
		for (int i = 0; i < in_dim; i++)
		{
			acc += in[i] * w[i * out_dim + o];
		}
		out[o] = acc;
	}
}

// any of din, dw and db can be NULL if you dont need them
static void matmul_backward(float *din, float *dw, float *db,
const float *dout, const float *in, const float *w,
int in_dim, int out_dim)
{
	for (int o = 0; o < out_dim; o++)
	{
		float d = dout[o];
		if (db) db[o] += d;
		for (int i = 0; i < in_dim; i++)
		{
			if (dw) dw[i * out_dim + o] += in[i] * d;
			if (din) din[i] += w[i * out_dim + o] * d;
		}
	}
}

// ==gelu==
// tanh approximation
static void gelu_forward(float *out, const float *x, int n)
{
	for (int i = 0; i < n; i++)
	{
		float v = x[i];
		out[i] = v * 0.5f * (1.0f + tanhf(GELU_SQ * (v + GELU_C * v * v * v)));
	}
}

static void gelu_backward(float *dx, const float *dout, const float *x, int n)
{
	for (int i = 0; i < n; i++)
	{
		float v = x[i];
		float inner = GELU_SQ * (v + GELU_C * v * v * v);
		float tanh_ = tanhf(inner);
		float sech2 = 1.0f - tanh_ * tanh_;
		float dtanh = GELU_SQ * (1.0f + 3.0f * GELU_C * v * v);
		dx[i] += dout[i] * (0.5f * (1.0f + tanh_) + v * 0.5f * sech2 * dtanh);
	}
}

// ==model structs==
typedef struct
{
	Parameter qkv_w;
	Parameter qkv_b;
	Parameter proj_w;
	Parameter proj_b;
} Attention;

typedef struct
{
	Parameter norm1_scale;
	Parameter norm1_bias;
	Parameter norm2_scale;
	Parameter norm2_bias;
	Parameter ff_up_w;
	Parameter ff_up_b;
	Parameter ff_down_w;
	Parameter ff_down_b;
	Attention attn;
} TransformerBlock;

typedef struct
{
	Parameter token_embedding;
	Parameter position_embedding;
	TransformerBlock *blocks;
	Parameter final_scale;
	Parameter final_bias;
	Parameter lm_head;
	int step;
} Model;

// activations saved during the forward pass so backward can use them
typedef struct
{
	float *qkv;
	float *attn_scores;
	float *attn_out;
	float *norm1_out;
	float norm1_mean[MAX_SEQ_LEN];
	float norm1_rstd[MAX_SEQ_LEN];
	float *attn_residual;
	float *norm2_out;
	float norm2_mean[MAX_SEQ_LEN];
	float norm2_rstd[MAX_SEQ_LEN];
	float *ff_pre;
	float *ff_post;
} BlockScratch;

static BlockScratch *block_scratch_alloc(int T)
{
	BlockScratch *s = malloc(sizeof(BlockScratch));
	int H = HIDDEN_DIM;
	int ff = FF_DIM;
	int nh = NUM_HEADS;
	s->qkv = malloc(T * 3 * H * sizeof(float));
	s->attn_scores = malloc(nh * T * T * sizeof(float));
	s->attn_out = malloc(T * H * sizeof(float));
	s->norm1_out = malloc(T * H * sizeof(float));
	s->attn_residual = malloc(T * H * sizeof(float));
	s->norm2_out = malloc(T * H * sizeof(float));
	s->ff_pre = malloc(T * ff * sizeof(float));
	s->ff_post = malloc(T * ff * sizeof(float));
	return s;
}

static void block_scratch_free(BlockScratch *s)
{
	free(s->qkv);
	free(s->attn_scores);
	free(s->attn_out);
	free(s->norm1_out);
	free(s->attn_residual);
	free(s->norm2_out);
	free(s->ff_pre);
	free(s->ff_post);
	free(s);
}

// ==attention==
static void attention_forward(float *out, const float *in, Attention *attn,
BlockScratch *scratch, int T)
{
	int H = HIDDEN_DIM;
	int nh = NUM_HEADS;
	int hd = HEAD_DIM;
	float scale = 1.0f / sqrtf((float)hd);

	// == project every token to q k v ==
	for (int t = 0; t < T; t++)
	{
		matmul_forward(scratch->qkv + t * 3 * H, in + t * H,
		attn->qkv_w.w, attn->qkv_b.w, H, 3 * H);
	}

	// q k v are stored one after the other
	const float *Q = scratch->qkv;
	const float *K = scratch->qkv + T * H;
	const float *V = scratch->qkv + 2 * T * H;
	memset(scratch->attn_out, 0, T * H * sizeof(float));

	// == run every head ==
	for (int h = 0; h < nh; h++)
	{
		int off = h * hd;
		float *scores = scratch->attn_scores + h * T * T;

		// scores = q dot k, with the causal mask then softmax
		for (int qi = 0; qi < T; qi++)
		{
			for (int ki = 0; ki < T; ki++)
			{
				float dot = 0.0f;
				for (int d = 0; d < hd; d++)
				{
					dot += Q[qi * H + off + d] * K[ki * H + off + d];
				}
				// cant look at the future
				scores[qi * T + ki] = (ki > qi) ? -1e9f : dot * scale;
			}
			softmax_inplace(scores + qi * T, T);
		}

		// weighted sum of the values
		for (int qi = 0; qi < T; qi++)
		{
			for (int d = 0; d < hd; d++)
			{
				float acc = 0.0f;
				for (int vi = 0; vi < T; vi++)
				{
					acc += scores[qi * T + vi] * V[vi * H + off + d];
				}
				scratch->attn_out[qi * H + off + d] = acc;
			}
		}
	}

	// == output projection ==
	for (int t = 0; t < T; t++)
	{
		matmul_forward(out + t * H, scratch->attn_out + t * H,
		attn->proj_w.w, attn->proj_b.w, H, H);
	}
}

static void attention_backward(float *din, Attention *attn, BlockScratch *scratch,
const float *dout, const float *in, int T)
{
	int H = HIDDEN_DIM;
	int nh = NUM_HEADS;
	int hd = HEAD_DIM;
	float scale = 1.0f / sqrtf((float)hd);

	float *d_attn_out = calloc(T * H, sizeof(float));
	float *d_qkv = calloc((size_t)T * 3 * H, sizeof(float));
	float *dQ = d_qkv;
	float *dK = d_qkv + T * H;
	float *dV = d_qkv + 2 * T * H;
	const float *Q = scratch->qkv;
	const float *K = scratch->qkv + T * H;
	const float *V = scratch->qkv + 2 * T * H;

	// == back through the output projection ==
	for (int t = 0; t < T; t++)
	{
		matmul_backward(d_attn_out + t * H, attn->proj_w.g, attn->proj_b.g,
		dout + t * H, scratch->attn_out + t * H,
		attn->proj_w.w, H, H);
	}

	// == back through every head ==
	for (int h = 0; h < nh; h++)
	{
		int off = h * hd;
		float *scores = scratch->attn_scores + h * T * T;
		float *d_scores = calloc(T * T, sizeof(float));

		// grads for the scores and for v
		for (int qi = 0; qi < T; qi++)
		{
			for (int d = 0; d < hd; d++)
			{
				float g = d_attn_out[qi * H + off + d];
				for (int vi = 0; vi < T; vi++)
				{
					d_scores[qi * T + vi] += g * V[vi * H + off + d];
					dV[vi * H + off + d] += scores[qi * T + vi] * g;
				}
			}
		}

		// back through the softmax then into q and k
		for (int qi = 0; qi < T; qi++)
		{
			float *row = scores + qi * T;
			float *drow = d_scores + qi * T;
			float dot = 0.0f;
			for (int ki = 0; ki < T; ki++) dot += drow[ki] * row[ki];
			// masked positions have no grad so stop at qi
			for (int ki = 0; ki <= qi; ki++)
			{
				float ds = row[ki] * (drow[ki] - dot) * scale;
				for (int d = 0; d < hd; d++)
				{
					dQ[qi * H + off + d] += ds * K[ki * H + off + d];
					dK[ki * H + off + d] += ds * Q[qi * H + off + d];
				}
			}
		}
		free(d_scores);
	}

	// == back through the qkv projection ==
	for (int t = 0; t < T; t++)
	{
		matmul_backward(din + t * H, attn->qkv_w.g, attn->qkv_b.g,
		d_qkv + t * 3 * H, in + t * H, attn->qkv_w.w, H, 3 * H);
	}

	free(d_attn_out);
	free(d_qkv);
}

// ==transformer block==
static void block_forward(float *out, const float *in, TransformerBlock *blk,
BlockScratch *scratch, int T)
{
	int H = HIDDEN_DIM;
	int ff = FF_DIM;

	// == norm 1 ==
	for (int t = 0; t < T; t++)
	{
		layer_norm_forward(scratch->norm1_out + t * H,
		&scratch->norm1_mean[t],
		&scratch->norm1_rstd[t],
		in + t * H,
		blk->norm1_scale.w,
		blk->norm1_bias.w,
		H);
	}

	// == attention + residual ==
	float *attn_out = calloc(T * H, sizeof(float));
	attention_forward(attn_out, scratch->norm1_out, &blk->attn, scratch, T);
	for (int i = 0; i < T * H; i++)
	{
		scratch->attn_residual[i] = in[i] + attn_out[i];
	}
	free(attn_out);

	// == norm 2 ==
	for (int t = 0; t < T; t++)
	{
		layer_norm_forward(scratch->norm2_out + t * H,
		&scratch->norm2_mean[t],
		&scratch->norm2_rstd[t],
		scratch->attn_residual + t * H,
		blk->norm2_scale.w,
		blk->norm2_bias.w,
		H);
	}

	// == feed forward + residual ==
	for (int t = 0; t < T; t++)
	{
		// up projection
		matmul_forward(scratch->ff_pre + t * ff,
		scratch->norm2_out + t * H,
		blk->ff_up_w.w,
		blk->ff_up_b.w,
		H, ff);

		// activation
		gelu_forward(scratch->ff_post + t * ff,
		scratch->ff_pre + t * ff,
		ff);

		// down projection
		float ff_out[HIDDEN_DIM];
		matmul_forward(ff_out,
		scratch->ff_post + t * ff,
		blk->ff_down_w.w,
		blk->ff_down_b.w,
		ff, H);

		// add the residual
		for (int i = 0; i < H; i++)
		{
			out[t * H + i] = scratch->attn_residual[t * H + i] + ff_out[i];
		}
	}
}

static void block_backward(float *din, TransformerBlock *blk, BlockScratch *scratch,
const float *dout, const float *in, int T)
{
	int H = HIDDEN_DIM;
	int ff = FF_DIM;

	float *d_resid2 = calloc(T * H, sizeof(float));
	float *d_norm2_out = calloc(T * H, sizeof(float));

	// == back through the feed forward ==
	for (int t = 0; t < T; t++)
	{
		// the residual passes the grad straight through
		for (int i = 0; i < H; i++) d_resid2[t * H + i] += dout[t * H + i];

		// down projection
		float d_ff_post[FF_DIM] = {0};
		matmul_backward(d_ff_post, blk->ff_down_w.g, blk->ff_down_b.g,
		dout + t * H, scratch->ff_post + t * ff,
		blk->ff_down_w.w, ff, H);

		// gelu
		float d_ff_pre[FF_DIM] = {0};
		gelu_backward(d_ff_pre, d_ff_post, scratch->ff_pre + t * ff, ff);

		// up projection
		matmul_backward(d_norm2_out + t * H, blk->ff_up_w.g, blk->ff_up_b.g,
		d_ff_pre, scratch->norm2_out + t * H,
		blk->ff_up_w.w, H, ff);
	}

	// == back through norm 2 ==
	float *d_attn_res = calloc(T * H, sizeof(float));
	for (int t = 0; t < T; t++)
	{
		layer_norm_backward(d_attn_res + t * H,
		blk->norm2_scale.g,
		blk->norm2_bias.g,
		d_norm2_out + t * H,
		scratch->attn_residual + t * H,
		blk->norm2_scale.w,
		scratch->norm2_mean[t],
		scratch->norm2_rstd[t],
		H);
	}
	free(d_norm2_out);

	// add the residual grad
	for (int i = 0; i < T * H; i++) d_attn_res[i] += d_resid2[i];
	free(d_resid2);

	// == back through attention ==
	float *d_norm1_out = calloc(T * H, sizeof(float));
	attention_backward(d_norm1_out, &blk->attn, scratch, d_attn_res, scratch->norm1_out, T);

	// residual again
	for (int i = 0; i < T * H; i++) din[i] += d_attn_res[i];
	free(d_attn_res);

	// == back through norm 1 ==
	for (int t = 0; t < T; t++)
	{
		layer_norm_backward(din + t * H,
		blk->norm1_scale.g,
		blk->norm1_bias.g,
		d_norm1_out + t * H,
		in + t * H,
		blk->norm1_scale.w,
		scratch->norm1_mean[t],
		scratch->norm1_rstd[t],
		H);
	}
	free(d_norm1_out);
}

// ==loss==
// cross entropy, also fills dlogits (the grad of the loss) while it is at it
static float cross_entropy(float *logits, const int *target, float *dlogits,
int B, int T, int V)
{
	float loss = 0.0f;
	float inv = 1.0f / (float)(B * T);
	for (int b = 0; b < B; b++)
	{
		for (int t = 0; t < T; t++)
		{
			int gt = target[b * T + t];
			if (gt == TOKEN_PAD) continue;
			float *row = logits + (b * T + t) * V;
			float *dr = dlogits + (b * T + t) * V;
			// max for stability
			float mx = row[0];
			for (int v = 1; v < V; v++) if (row[v] > mx) mx = row[v];
			float sum_exp = 0.0f;
			for (int v = 0; v < V; v++) sum_exp += expf(row[v] - mx);
			loss += mx + logf(sum_exp) - row[gt];
			// grad = softmax - one hot
			for (int v = 0; v < V; v++)
			{
				dr[v] = (expf(row[v] - mx) / sum_exp - (v == gt ? 1.0f : 0.0f)) * inv;
			}
		}
	}
	return loss * inv;
}

// ==model lifecycle==
static Model *model_new(void)
{
	Model *m = calloc(1, sizeof(Model));
	// == embeddings ==
	parameter_alloc(&m->token_embedding, VOCAB_SIZE * HIDDEN_DIM);
	parameter_kaiming(&m->token_embedding, VOCAB_SIZE);
	parameter_alloc(&m->position_embedding, MAX_SEQ_LEN * HIDDEN_DIM);
	parameter_kaiming(&m->position_embedding, MAX_SEQ_LEN);
	// == transformer blocks ==
	m->blocks = malloc(NUM_LAYERS * sizeof(TransformerBlock));
	for (int l = 0; l < NUM_LAYERS; l++)
	{
		TransformerBlock *b = &m->blocks[l];
		// attention
		parameter_alloc(&b->attn.qkv_w, HIDDEN_DIM * 3 * HIDDEN_DIM);
		parameter_kaiming(&b->attn.qkv_w, HIDDEN_DIM);
		parameter_alloc(&b->attn.qkv_b, 3 * HIDDEN_DIM);
		parameter_alloc(&b->attn.proj_w, HIDDEN_DIM * HIDDEN_DIM);
		parameter_kaiming(&b->attn.proj_w, HIDDEN_DIM);
		parameter_alloc(&b->attn.proj_b, HIDDEN_DIM);
		// layer norms (scale starts at 1, bias at 0)
		parameter_alloc(&b->norm1_scale, HIDDEN_DIM);
		parameter_ones(&b->norm1_scale);
		parameter_alloc(&b->norm1_bias, HIDDEN_DIM);
		parameter_alloc(&b->norm2_scale, HIDDEN_DIM);
		parameter_ones(&b->norm2_scale);
		parameter_alloc(&b->norm2_bias, HIDDEN_DIM);
		// feed forward
		parameter_alloc(&b->ff_up_w, HIDDEN_DIM * FF_DIM);
		parameter_kaiming(&b->ff_up_w, HIDDEN_DIM);
		parameter_alloc(&b->ff_up_b, FF_DIM);
		parameter_alloc(&b->ff_down_w, FF_DIM * HIDDEN_DIM);
		parameter_kaiming(&b->ff_down_w, FF_DIM);
		parameter_alloc(&b->ff_down_b, HIDDEN_DIM);
	}
	// == final norm and output head ==
	parameter_alloc(&m->final_scale, HIDDEN_DIM);
	parameter_ones(&m->final_scale);
	parameter_alloc(&m->final_bias, HIDDEN_DIM);
	parameter_alloc(&m->lm_head, HIDDEN_DIM * VOCAB_SIZE);
	parameter_kaiming(&m->lm_head, HIDDEN_DIM);
	// == print the size ==
	long params = (long)VOCAB_SIZE * HIDDEN_DIM + MAX_SEQ_LEN * HIDDEN_DIM +
	(long)NUM_LAYERS * (HIDDEN_DIM * 3 * HIDDEN_DIM + HIDDEN_DIM * HIDDEN_DIM +
	HIDDEN_DIM * FF_DIM + FF_DIM * HIDDEN_DIM + 4 * HIDDEN_DIM) +
	2 * HIDDEN_DIM + (long)HIDDEN_DIM * VOCAB_SIZE;
	// x4 floats per param (w g m v) x4 bytes each
	printf("params %.2fm using %ld mb of ram hope you have enough\n\n", (double)params / 1e6, params * 4 * 4 / 1000000);
	return m;
}

static void model_free(Model *m)
{
	parameter_free(&m->token_embedding);
	parameter_free(&m->position_embedding);
	for (int l = 0; l < NUM_LAYERS; l++)
	{
		TransformerBlock *b = &m->blocks[l];
		parameter_free(&b->attn.qkv_w);
		parameter_free(&b->attn.qkv_b);
		parameter_free(&b->attn.proj_w);
		parameter_free(&b->attn.proj_b);
		parameter_free(&b->norm1_scale);
		parameter_free(&b->norm1_bias);
		parameter_free(&b->norm2_scale);
		parameter_free(&b->norm2_bias);
		parameter_free(&b->ff_up_w);
		parameter_free(&b->ff_up_b);
		parameter_free(&b->ff_down_w);
		parameter_free(&b->ff_down_b);
	}
	free(m->blocks);
	parameter_free(&m->final_scale);
	parameter_free(&m->final_bias);
	parameter_free(&m->lm_head);
	free(m);
}

// ==checkpoints==
// writes to a .tmp file first then renames so a crash cant corrupt the old checkpoint
static void model_save(Model *m, const char *path)
{
	char tmp[512];
	snprintf(tmp, sizeof(tmp), "%s.tmp", path);
	FILE *f = fopen(tmp, "wb");
	if (!f)
	{
		fprintf(stderr, "cannot write %s disk full or the gremlins ate it\n", tmp);
		return;
	}
	fwrite(&m->step, sizeof(int), 1, f);
	// each param saves its weights and both adam moments
	#define WRITE_PARAM(p) \
	fwrite((p).w, sizeof(float), (p).size, f); \
	fwrite((p).m, sizeof(float), (p).size, f); \
	fwrite((p).v, sizeof(float), (p).size, f)
	WRITE_PARAM(m->token_embedding);
	WRITE_PARAM(m->position_embedding);
	for (int l = 0; l < NUM_LAYERS; l++)
	{
		TransformerBlock *b = &m->blocks[l];
		WRITE_PARAM(b->attn.qkv_w);
		WRITE_PARAM(b->attn.qkv_b);
		WRITE_PARAM(b->attn.proj_w);
		WRITE_PARAM(b->attn.proj_b);
		WRITE_PARAM(b->norm1_scale);
		WRITE_PARAM(b->norm1_bias);
		WRITE_PARAM(b->norm2_scale);
		WRITE_PARAM(b->norm2_bias);
		WRITE_PARAM(b->ff_up_w);
		WRITE_PARAM(b->ff_up_b);
		WRITE_PARAM(b->ff_down_w);
		WRITE_PARAM(b->ff_down_b);
	}
	WRITE_PARAM(m->final_scale);
	WRITE_PARAM(m->final_bias);
	WRITE_PARAM(m->lm_head);
	#undef WRITE_PARAM
	fclose(f);
	rename(tmp, path);
	printf("saved %s step %d phew\n", path, m->step);
}

// must read in the exact same order model_save wrote
static Model *model_load(const char *path)
{
	FILE *f = fopen(path, "rb");
	if (!f)
	{
		fprintf(stderr, "cannot load %s, starting fresh like a newborn baby\n\n", path);
		return NULL;
	}
	Model *m = model_new();
	fread(&m->step, sizeof(int), 1, f);
	#define READ_PARAM(p) \
	fread((p).w, sizeof(float), (p).size, f); \
	fread((p).m, sizeof(float), (p).size, f); \
	fread((p).v, sizeof(float), (p).size, f)
	READ_PARAM(m->token_embedding);
	READ_PARAM(m->position_embedding);
	for (int l = 0; l < NUM_LAYERS; l++)
	{
		TransformerBlock *b = &m->blocks[l];
		READ_PARAM(b->attn.qkv_w);
		READ_PARAM(b->attn.qkv_b);
		READ_PARAM(b->attn.proj_w);
		READ_PARAM(b->attn.proj_b);
		READ_PARAM(b->norm1_scale);
		READ_PARAM(b->norm1_bias);
		READ_PARAM(b->norm2_scale);
		READ_PARAM(b->norm2_bias);
		READ_PARAM(b->ff_up_w);
		READ_PARAM(b->ff_up_b);
		READ_PARAM(b->ff_down_w);
		READ_PARAM(b->ff_down_b);
	}
	READ_PARAM(m->final_scale);
	READ_PARAM(m->final_bias);
	READ_PARAM(m->lm_head);
	#undef READ_PARAM
	fclose(f);
	printf("loaded %s from step %d, lets continue the madness\n\n", path, m->step);
	return m;
}

// ==optimizer==
// scale all grads down if the global norm is too big
static void clip_gradients(Model *m)
{
	// == add up the squared grads ==
	float total = 0.0f;
	#define GRAD_NORM(p) \
	for (int _i = 0; _i < (p).size; _i++) total += (p).g[_i] * (p).g[_i]
	GRAD_NORM(m->token_embedding);
	GRAD_NORM(m->position_embedding);
	for (int l = 0; l < NUM_LAYERS; l++)
	{
		TransformerBlock *b = &m->blocks[l];
		GRAD_NORM(b->attn.qkv_w);
		GRAD_NORM(b->attn.qkv_b);
		GRAD_NORM(b->attn.proj_w);
		GRAD_NORM(b->attn.proj_b);
		GRAD_NORM(b->norm1_scale);
		GRAD_NORM(b->norm1_bias);
		GRAD_NORM(b->norm2_scale);
		GRAD_NORM(b->norm2_bias);
		GRAD_NORM(b->ff_up_w);
		GRAD_NORM(b->ff_up_b);
		GRAD_NORM(b->ff_down_w);
		GRAD_NORM(b->ff_down_b);
	}
	GRAD_NORM(m->final_scale);
	GRAD_NORM(m->final_bias);
	GRAD_NORM(m->lm_head);
	#undef GRAD_NORM
	float norm = sqrtf(total);
	// small enough, nothing to do
	if (norm <= CLIP_GRAD_NORM) return;
	// == shrink every grad ==
	float scale = CLIP_GRAD_NORM / norm;
	#define SCALE_GRAD(p) \
	for (int _i = 0; _i < (p).size; _i++) (p).g[_i] *= scale
	SCALE_GRAD(m->token_embedding);
	SCALE_GRAD(m->position_embedding);
	for (int l = 0; l < NUM_LAYERS; l++)
	{
		TransformerBlock *b = &m->blocks[l];
		SCALE_GRAD(b->attn.qkv_w);
		SCALE_GRAD(b->attn.qkv_b);
		SCALE_GRAD(b->attn.proj_w);
		SCALE_GRAD(b->attn.proj_b);
		SCALE_GRAD(b->norm1_scale);
		SCALE_GRAD(b->norm1_bias);
		SCALE_GRAD(b->norm2_scale);
		SCALE_GRAD(b->norm2_bias);
		SCALE_GRAD(b->ff_up_w);
		SCALE_GRAD(b->ff_up_b);
		SCALE_GRAD(b->ff_down_w);
		SCALE_GRAD(b->ff_down_b);
	}
	SCALE_GRAD(m->final_scale);
	SCALE_GRAD(m->final_bias);
	SCALE_GRAD(m->lm_head);
	#undef SCALE_GRAD
}

// adam update for one tensor, also zeroes the grads after using them
static void adam_update_parameter(Parameter *p, float lr, int step)
{
	float bias_correction1 = 1.0f - powf(ADAM_BETA1, (float)step);
	float bias_correction2 = 1.0f - powf(ADAM_BETA2, (float)step);
	for (int i = 0; i < p->size; i++)
	{
		float g = p->g[i];
		// update the moments
		p->m[i] = ADAM_BETA1 * p->m[i] + (1.0f - ADAM_BETA1) * g;
		p->v[i] = ADAM_BETA2 * p->v[i] + (1.0f - ADAM_BETA2) * g * g;
		// update the weight
		p->w[i] -= lr * (p->m[i] / bias_correction1) /
		(sqrtf(p->v[i] / bias_correction2) + ADAM_EPS);
		// reset the grad for the next step
		p->g[i] = 0.0f;
	}
}

static void adam_step(Model *m, float lr)
{
	int step = ++m->step;
	#define ADAM_UPDATE(p) adam_update_parameter(&(p), lr, step)
	ADAM_UPDATE(m->token_embedding);
	ADAM_UPDATE(m->position_embedding);
	for (int l = 0; l < NUM_LAYERS; l++)
	{
		TransformerBlock *b = &m->blocks[l];
		ADAM_UPDATE(b->attn.qkv_w);
		ADAM_UPDATE(b->attn.qkv_b);
		ADAM_UPDATE(b->attn.proj_w);
		ADAM_UPDATE(b->attn.proj_b);
		ADAM_UPDATE(b->norm1_scale);
		ADAM_UPDATE(b->norm1_bias);
		ADAM_UPDATE(b->norm2_scale);
		ADAM_UPDATE(b->norm2_bias);
		ADAM_UPDATE(b->ff_up_w);
		ADAM_UPDATE(b->ff_up_b);
		ADAM_UPDATE(b->ff_down_w);
		ADAM_UPDATE(b->ff_down_b);
	}
	ADAM_UPDATE(m->final_scale);
	ADAM_UPDATE(m->final_bias);
	ADAM_UPDATE(m->lm_head);
	#undef ADAM_UPDATE
}

// ==training==
// everything one thread needs to run its part of the batch
typedef struct
{
	Model *model;
	int *input;
	int *target;
	int B;
	int T;
	float loss;
} ThreadWork;

static void forward_backward(ThreadWork *work)
{
	Model *m = work->model;
	int B = work->B;
	int T = work->T;
	int H = HIDDEN_DIM;
	int V = VOCAB_SIZE;
	int L = NUM_LAYERS;
	int BT = B * T;

	float **layer_in = malloc((L + 1) * sizeof(float *));
	BlockScratch **scratch = malloc(L * sizeof(BlockScratch *));
	for (int l = 0; l < L; l++) scratch[l] = block_scratch_alloc(T);

	// ==forward==
	// == embeddings (token + position) ==
	layer_in[0] = malloc(BT * H * sizeof(float));
	for (int b = 0; b < B; b++)
	{
		for (int t = 0; t < T; t++)
		{
			int tok = work->input[b * T + t];
			float *row = layer_in[0] + (b * T + t) * H;
			for (int i = 0; i < H; i++)
			{
				row[i] = m->token_embedding.w[tok * H + i] + m->position_embedding.w[t * H + i];
			}
		}
	}

	// == transformer blocks ==
	for (int l = 0; l < L; l++)
	{
		layer_in[l + 1] = malloc(BT * H * sizeof(float));
		for (int b = 0; b < B; b++)
		{
			block_forward(layer_in[l + 1] + b * T * H,
			layer_in[l] + b * T * H,
			&m->blocks[l],
			scratch[l],
			T);
		}
	}

	// == final norm ==
	float *final_norm_mean = malloc(BT * sizeof(float));
	float *final_norm_rstd = malloc(BT * sizeof(float));
	float *final_norm_out = malloc(BT * H * sizeof(float));
	for (int bt = 0; bt < BT; bt++)
	{
		layer_norm_forward(final_norm_out + bt * H,
		&final_norm_mean[bt],
		&final_norm_rstd[bt],
		layer_in[L] + bt * H,
		m->final_scale.w,
		m->final_bias.w,
		H);
	}

	// == logits ==
	float *logits = malloc(BT * V * sizeof(float));
	for (int bt = 0; bt < BT; bt++)
	{
		matmul_forward(logits + bt * V,
		final_norm_out + bt * H,
		m->lm_head.w,
		NULL,
		H, V);
	}

	// == loss ==
	float *dlogits = calloc(BT * V, sizeof(float));
	work->loss = cross_entropy(logits, work->target, dlogits, B, T, V);
	free(logits);

	// ==backward==
	// == lm head ==
	float *d_final_norm_out = calloc(BT * H, sizeof(float));
	for (int bt = 0; bt < BT; bt++)
	{
		matmul_backward(d_final_norm_out + bt * H,
		m->lm_head.g,
		NULL,
		dlogits + bt * V,
		final_norm_out + bt * H,
		m->lm_head.w,
		H, V);
	}
	free(dlogits);
	free(final_norm_out);

	// == final norm ==
	float *dlayer = calloc(BT * H, sizeof(float));
	for (int bt = 0; bt < BT; bt++)
	{
		layer_norm_backward(dlayer + bt * H,
		m->final_scale.g,
		m->final_bias.g,
		d_final_norm_out + bt * H,
		layer_in[L] + bt * H,
		m->final_scale.w,
		final_norm_mean[bt],
		final_norm_rstd[bt],
		H);
	}
	free(d_final_norm_out);
	free(final_norm_mean);
	free(final_norm_rstd);

	// == blocks, last to first ==
	for (int l = L - 1; l >= 0; l--)
	{
		float *din = calloc(BT * H, sizeof(float));
		for (int b = 0; b < B; b++)
		{
			block_backward(din + b * T * H,
			&m->blocks[l],
			scratch[l],
			dlayer + b * T * H,
			layer_in[l] + b * T * H,
			T);
		}
		free(dlayer);
		dlayer = din;
	}

	// == embeddings ==
	for (int b = 0; b < B; b++)
	{
		for (int t = 0; t < T; t++)
		{
			int tok = work->input[b * T + t];
			float *drow = dlayer + (b * T + t) * H;
			for (int i = 0; i < H; i++)
			{
				m->token_embedding.g[tok * H + i] += drow[i];
				m->position_embedding.g[t * H + i] += drow[i];
			}
		}
	}
	free(dlayer);

	// ==cleanup==
	for (int l = 0; l <= L; l++) free(layer_in[l]);
	free(layer_in);
	for (int l = 0; l < L; l++) block_scratch_free(scratch[l]);
	free(scratch);
}

// pthread wrapper
static void *thread_forward_backward(void *arg)
{
	forward_backward((ThreadWork *)arg);
	return NULL;
}

// one training step, splits the batch between 2 threads
static float train_step(Model *m, int *input, int *target, int B, int T)
{
	// cant split a batch of 1
	if (B == 1)
	{
		ThreadWork work = {m, input, target, 1, T, 0.0f};
		forward_backward(&work);
		return work.loss;
	}

	// == split the batch in half ==
	int half = B / 2;
	ThreadWork w0 = {m, input, target, half, T, 0.0f};
	ThreadWork w1 = {m, input + half * T, target + half * T, B - half, T, 0.0f};
	pthread_t t0, t1;
	pthread_create(&t0, NULL, thread_forward_backward, &w0);
	pthread_create(&t1, NULL, thread_forward_backward, &w1);
	pthread_join(t0, NULL);
	pthread_join(t1, NULL);
	// average of both losses
	return (w0.loss + w1.loss) * 0.5f;
}

// ==sampling==
// pick the next token from the probabilities
static int sample_token(float *probs, int V, float temperature, int top_k)
{
	// temp 0 = greedy, just take the best one
	if (temperature <= 0.0f)
	{
		int best = 0;
		for (int v = 1; v < V; v++) if (probs[v] > probs[best]) best = v;
		return best;
	}

	// == apply temperature ==
	for (int v = 0; v < V; v++)
	{
		probs[v] = expf(logf(probs[v] + 1e-10f) / temperature);
	}

	// == top k, zero everything below the k-th best ==
	if (top_k > 0 && top_k < V)
	{
		float *temp = malloc(V * sizeof(float));
		memcpy(temp, probs, V * sizeof(float));
		for (int i = 0; i < top_k; i++)
		{
			float max_val = -1e9f;
			int max_idx = 0;
			for (int j = 0; j < V; j++)
			{
				if (temp[j] > max_val)
				{
					max_val = temp[j];
					max_idx = j;
				}
			}
			// the k-th best value is the cutoff
			if (i == top_k - 1)
			{
				float threshold = max_val;
				for (int j = 0; j < V; j++)
				{
					if (probs[j] < threshold) probs[j] = 0.0f;
				}
			}
			temp[max_idx] = -1e9f;
		}
		free(temp);
	}

	// == roll the dice ==
	float sum = 0.0f;
	for (int v = 0; v < V; v++) sum += probs[v];
	float r = (float)rand() / (float)RAND_MAX;
	float cum = 0.0f;
	for (int v = 0; v < V; v++)
	{
		cum += probs[v] / sum;
		if (r < cum) return v;
	}
	return V - 1;
}

// ==generation==
// generate num_gen tokens after the prompt and print them as they come
static void generate(Model *m, BpeVocab *vocab, int *prompt, int prompt_len,
int num_gen, float temperature)
{
	int H = HIDDEN_DIM;
	int V = VOCAB_SIZE;
	int L = NUM_LAYERS;
	int context[MAX_SEQ_LEN];
	int cur_len = prompt_len < MAX_SEQ_LEN ? prompt_len : MAX_SEQ_LEN;
	memcpy(context, prompt, cur_len * sizeof(int));

	float *row = malloc(V * sizeof(float));
	BlockScratch **scratch = malloc(L * sizeof(BlockScratch *));
	for (int l = 0; l < L; l++) scratch[l] = block_scratch_alloc(MAX_SEQ_LEN);

	// == one token per loop ==
	for (int g = 0; g < num_gen; g++)
	{
		int cur = cur_len + g;
		// context is full, slide the window by 1
		if (cur >= MAX_SEQ_LEN)
		{
			memmove(context, context + 1, (MAX_SEQ_LEN - 1) * sizeof(int));
			cur = MAX_SEQ_LEN - 1;
		}

		// == embeddings ==
		float *x = malloc(cur * H * sizeof(float));
		for (int t = 0; t < cur; t++)
		{
			int tok = context[t];
			for (int i = 0; i < H; i++)
			{
				x[t * H + i] = m->token_embedding.w[tok * H + i] + m->position_embedding.w[t * H + i];
			}
		}

		// == blocks ==
		for (int l = 0; l < L; l++)
		{
			float *nx = malloc(cur * H * sizeof(float));
			block_forward(nx, x, &m->blocks[l], scratch[l], cur);
			free(x);
			x = nx;
		}

		// == final norm + logits for the last position only ==
		float normed[HIDDEN_DIM];
		float mean, rstd;
		layer_norm_forward(normed, &mean, &rstd,
		x + (cur - 1) * H,
		m->final_scale.w,
		m->final_bias.w,
		H);
		matmul_forward(row, normed, m->lm_head.w, NULL, H, V);

		// never generate unk or pad
		row[TOKEN_UNK] = -1e9f;
		row[TOKEN_PAD] = -1e9f;

		// == repetition penalty on recent tokens ==
		int start = cur_len > REP_PENALTY_WINDOW ? cur_len - REP_PENALTY_WINDOW : 0;
		for (int i = start; i < cur_len; i++)
		{
			int tok = context[i];
			row[tok] *= REP_PENALTY_FACTOR;
		}

		softmax_inplace(row, V);

		// == pick the next token ==
		int next = sample_token(row, V, temperature, TOP_K);
		free(x);
		// end of sentence, stop
		if (next == TOKEN_EOS) break;
		context[cur] = next;

		// == print it ==
		if (vocab && next < vocab->size)
		{
			const char *p = vocab->tokens[next].piece;

			// the word start marker becomes a space
			if ((unsigned char)p[0] == 0xe2 &&
			(unsigned char)p[1] == 0x96 &&
			(unsigned char)p[2] == 0x81)
			{
				printf(" %s", p + 3);
			} else
			{
				printf("%s", p);
			}
		} else
		{
			// no vocab, print the raw id
			printf("[%d]", next);
		}
		fflush(stdout);
	}
	printf("\n");
	for (int l = 0; l < L; l++) block_scratch_free(scratch[l]);
	free(scratch);
	free(row);
}

// ==chat==
static void chat_loop(Model *m, BpeVocab *vocab)
{
	// ==system prompt==
	const char *system_prompt =
	"you are beb0, a helpful and friendly ai assistant created by abanoub. "
	"you answer questions, help with coding, and keep conversations light and fun. "
	"you are knowledgeable about tech, ai, and programming, but you explain things simply. "
	"you never use emojis and keep responses concise. you are running in a terminal.";

	int system_tokens[MAX_SEQ_LEN];
	int system_len = vocab_encode(vocab, system_prompt, system_tokens, MAX_SEQ_LEN);
	if (system_len >= MAX_SEQ_LEN) system_len = MAX_SEQ_LEN - 1;

	// ==vars==
	// history starts with the system prompt
	int history[MAX_SEQ_LEN * 4];
	int hist_len = 0;
	for (int i = 0; i < system_len; i++) history[hist_len++] = system_tokens[i];

	char line[2048];
	int user_tokens[MAX_SEQ_LEN];
	int prompt[MAX_SEQ_LEN];

	// ==banner==
	printf("\n\n");
	printf("\033[31mBBBBB    \033[32mEEEEEE  \033[31mBBBBB     \033[33m0000 \033[0m\n");
	printf("\033[31mBBBBBB   \033[32mEEEEEE  \033[31mBBBBB    \033[33m000000\033[0m\n");
	printf("\033[31mBB   BB  \033[32mEE      \033[31mBB   BB  \033[33m00  00\033[0m\n");
	printf("\033[31mBBBBBB   \033[32mEEEEEE  \033[31mBBBBBB   \033[33m00  00\033[0m\n");
	printf("\033[31mBBBBBB   \033[32mEEEEEE  \033[31mBBBBBB   \033[33m00  00\033[0m\n");
	printf("\033[31mBB   BB  \033[32mEE      \033[31mBB   BB  \033[33m00  00\033[0m\n");
	printf("\033[31mBBBBBB   \033[32mEEEEEE  \033[31mBBBBBB   \033[33m000000\033[0m\n");
	printf("\033[31mBBBBB    \033[32mEEEEEE  \033[31mBBBBB     \033[33m0000 \033[0m\n\n");
	printf("chat mode activated type /quit to /exit or /help for commands or /joke for a laugh\n\n");

	// ==jokes==
	const char *jokes[] = {
		"why did the ai cross the road to get to the other side of the dataset",
		"what do you call a neural net that sings a tune net",
		"why dont scientists trust atoms because they make up everything",
		"how many programmers does it take to change a light bulb none thats a hardware problem",
		"whats an ais favorite music algorythm",
		"why did the transformr go to therapy it had too many attentoin issues",
		"why do programmers prefer dark mode because light attracts bugs",
		"what do you call a computer that sings a dell",
		"why was the javascript developer sad because he didnt know how to 'null' his feelings",
		"how do you comfort a javascript bug you console it",
		"why did the developer go broke because he used up all his cache",
		"whats a computers favorite beat a hard drive",
		"why do java developers wear glasses because they cant C",
		"what do you call 8 hobbits hobbits hobbits",
		"why did the ai break up with the neural network it had too many layers of issues",
		"what do you get when you cross a snowman and a vampire frostbite",
		"why dont programmers like nature it has too many bugs",
		"what do you call a bear with no teeth a gummy bear",
		"why did the scarecrow win an award because he was outstanding in his field",
		"what do you call a fake noodle an impasta",
		"why cant you trust an atom they make up everything",
		"why did the bicycle fall over because it was two tired",
		"how does a penguin build its house igloos it together",
		"why cant you give elsa a balloon because she will let it go",
		"what do you call a fish with no eyes a fsh",
		"why did the math book look so sad because it had too many problems",
		"what do you call a belt made of watches a waist of time",
		"why did the coffee file a police report it got mugged",
		"what do you call a line of rabbits jumping backwards a receding hare line",
		"why did the ai stop playing chess because it kept getting checkmated by a human",
		"how many AI researchers does it take to change a light bulb none they just adjust the learning rate until it lights up"
	};
	int num_jokes = sizeof(jokes) / sizeof(jokes[0]);

	// ==main loop==
	while (!g_stop)
	{
		// == take input ==
		printf("\x1b[31mU =>> \x1b[0m");
		fflush(stdout);
		if (!fgets(line, sizeof(line), stdin)) break;
		line[strcspn(line, "\n")] = '\0';

		// == commands ==
		if (strcmp(line, "/quit") == 0 || strcmp(line, "/exit") == 0) break;
		else if (strcmp(line, "/help") == 0)
		{
			printf("\nHELP MENU as if you didnt know\n");
			printf("/exit /quit   - get outta here\n");
			printf("/restart       - start over like a phoenix\n");
			printf("/clear         - clear the screen magic\n");
			printf("/joke          - hear a terrible joke\n");
			printf("/help          - youre looking at it\n\n");
			continue;
		} else if (strcmp(line, "/joke") == 0)
		{
			// pick a random joke
			int idx = rand() % num_jokes;
			printf("\x1b[33mai =>> \x1b[0m%s\n", jokes[idx]);
			continue;
		} else if (strcmp(line, "/restart") == 0)
		{
			// clear the screen then replace this process with a fresh one
			printf("\033[H\033[2J");
			char *argv[] = { "./beb0", NULL };
			execv("/proc/self/exe", argv);
			perror("execv failed oops");
			exit(1);
		} else if (strcmp(line, "") == 0) continue;
		else if (strcmp(line, "/clear") == 0)
		{
			printf("\033[H\033[2J");
			continue;
		}

		// == encode the user message and add it to the history ==
		char user_line[2060];
		snprintf(user_line, sizeof(user_line), "user =>> %s\n", line);
		int user_len = vocab_encode(vocab, user_line, user_tokens, MAX_SEQ_LEN);
		if (user_len > MAX_SEQ_LEN) user_len = MAX_SEQ_LEN;

		for (int i = 0; i < user_len && hist_len < MAX_SEQ_LEN * 4; i++)
		{
			history[hist_len++] = user_tokens[i];
		}

		// == build the prompt from the last MAX_SEQ_LEN tokens ==
		int prompt_len = 0;
		int start = hist_len > MAX_SEQ_LEN ? hist_len - MAX_SEQ_LEN : 0;
		for (int i = start; i < hist_len; i++)
		{
			prompt[prompt_len++] = history[i];
		}

		// == generate the reply ==
		printf("\x1b[33mai =>> \x1b[0m");
		fflush(stdout);
		generate(m, vocab, prompt, prompt_len, 200, 0.5f);
	}
}

// ==main==
int main(int argc, char **argv)
{
	// ==setup==
	signal(SIGINT, handle_signal);
	signal(SIGTERM, handle_signal);
	srand((unsigned)time(NULL));

	BpeVocab *vocab = vocab_load("data/vocab.txt");

	// ==chat mode==
	// no args = just chat
	if (argc == 1)
	{
		Model *m = model_load("checkpoints/checkpoint.bin");
		if (!m)
		{
			printf("no checkpoint found creating a fresh model like a blank canvas\n");
			m = model_new();
		}
		chat_loop(m, vocab);
		if (vocab) vocab_free(vocab);
		model_free(m);
		return 0;
	}

	// ==train mode==
	// first arg is the corpus path
	const char *corpus_path = argv[1];
	Corpus corpus = corpus_load(corpus_path);
	if (!corpus.tokens)
	{
		fprintf(stderr, "failed to load corpus %s maybe its playing hide and seek\n", corpus_path);
		if (vocab) vocab_free(vocab);
		return 1;
	}

	// resume from the checkpoint if there is one
	Model *m = model_load("checkpoints/checkpoint.bin");
	if (!m)
	{
		printf("no checkpoint found creating a fresh model like a blank canvas  \n");
		m = model_new();
	}

	// ==vars==
	int B = PHYSICAL_BATCH;
	int T = MAX_SEQ_LEN;
	int *input = malloc(B * T * sizeof(int));
	int *target = malloc(B * T * sizeof(int));
	float lr = LEARNING_RATE;
	float accum_loss = 0.0f;

	printf("alright lets train this thing on %s for %d steps with batch %d and accumulation %d lr=%.1e... buckle up\n",
	corpus_path, NUM_STEPS, B, GRAD_ACCUM_STEPS, (double)lr);
	printf("thermal warn=%dc crit=%dc steamroller dualcore mode  i hope it doesnt melt\n",
	THERMAL_WARN_C, THERMAL_CRIT_C);

	// ==training loop==
	for (int step = 0; step < NUM_STEPS && !g_stop; step++)
	{
		// == accumulate grads over several mini batches ==
		accum_loss = 0.0f;
		for (int acc = 0; acc < GRAD_ACCUM_STEPS; acc++)
		{
			corpus_batch(&corpus, input, target, B, T);
			accum_loss += train_step(m, input, target, B, T);
		}
		accum_loss /= (float)GRAD_ACCUM_STEPS;

		// == update the weights ==
		clip_gradients(m);
		adam_step(m, lr);

		// == logging + thermal check ==
		if (step % LOG_EVERY == 0)
		{
			printf("step %5d, loss %.4f [is that good], lr %.2e, cpu %dc\n",
			step, accum_loss, (double)lr, cpu_temperature_c());
			fflush(stdout);
			thermal_check(&lr);
		}

		// == lr drops to the minimum after 50k steps ==
		if (m->step >= 50000) lr = MIN_LEARNING_RATE;

		// == save every now and then ==
		if (step % SAVE_EVERY == 0 && step > 0)
		{
			model_save(m, "checkpoints/checkpoint.bin");
		}
	}

	// ==cleanup==
	model_save(m, "checkpoints/checkpoint.bin");
	if (g_stop) printf("\nwhoa interrupted but i saved checkpoint youre welcome\n");

	corpus_free(&corpus);
	if (vocab) vocab_free(vocab);
	model_free(m);
	free(input);
	free(target);
	return 0;
}
