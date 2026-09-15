FROM ubuntu:24.04
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates git zstd build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN case "$TARGETARCH" in \
      arm64) asset=lean-4.27.0-linux_aarch64.tar.zst; digest=b256eec276baaaccc3eb3fa64d7ccff64f710b7caa074f305ba95e0013ad31e7 ;; \
      amd64) asset=lean-4.27.0-linux.tar.zst; digest=056e2dc8564fc064a801e69f3eb18c044b9b546bc8b0e5a2c00247f8a1cb8ce6 ;; \
      *) exit 1 ;; esac \
    && curl -fL --retry 3 "https://github.com/leanprover/lean4/releases/download/v4.27.0/$asset" -o /tmp/lean.tar.zst \
    && echo "$digest  /tmp/lean.tar.zst" | sha256sum -c - \
    && mkdir -p /opt/lean \
    && tar --zstd -xf /tmp/lean.tar.zst --strip-components=1 -C /opt/lean \
    && rm /tmp/lean.tar.zst
ENV PATH="/opt/lean/bin:${PATH}"
WORKDIR /opt/putnam
COPY lakefile.lean lake-manifest.json lean-toolchain ./
RUN lake exe cache get
# Lake consults Git origins even for an offline `lake env lean`. These immutable
# repositories are built as root and read by the unprivileged task user.
RUN git config --system --add safe.directory /opt/putnam \
    && for dep in /opt/putnam/.lake/packages/*; do git config --system --add safe.directory "$dep"; done
