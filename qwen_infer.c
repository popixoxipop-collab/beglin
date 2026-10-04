    } else if (W->kind == K_QNG64) {
        qng64_matvec_f32(W->packed, W->scales, W->bits, x, bias, y, W->out, W->in);    } else if (W->kind == K_QNG64) {
        qng64_matmul_f32(W->packed, W->scales, W->bits, x, bias, y, W->out, W->in, M);

