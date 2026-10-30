export const config = (() => {
  const prompt =
    "A small red sailboat glides across a calm lake at sunrise. Gentle waves lap against the hull and birds chirp in the distance.";
  const defaults = {
    hw: "b200",
    nodes: 1,
    gpus_per_node: 1,
    topology_mode: "auto",
    tp_size: 1,
    ulysses_degree: 1,
    ring_degree: 1,
    mode: "t2va",
    execution: "auto",
    resolution: "480",
    duration: "5",
    aspect: "16:9",
  };
  const topologyFor = (s) => ({
    tp_size: s.hw === "h100" && Number(s.gpus_per_node) > 1 ? 2 : 1,
    ulysses_degree:
      s.hw === "h100" && Number(s.gpus_per_node) > 1
        ? Number(s.gpus_per_node) / 2
        : Number(s.gpus_per_node),
    ring_degree: 1,
  });
  const config = {
    modelName: "Pika H3 Flex",
    supportedHardware: ["b200", "h200", "h100"],
    groupHardware: false,
    matchDims: [],
    overlayDims: [
      {
        id: "execution",
        title: "Execution",
        scope: "serve",
        default: "auto",
        description:
          "Automatic uses the measured B200 recipe for text generation at 480p/5 seconds on one or four GPUs. Other shapes and tasks use eager execution.",
        learnMore: "/cookbook/diffusion/MiniMax/Pika-H3-Flex#b200-measurements",
        options: [
          {
            id: "auto",
            label: "Automatic",
            description:
              "Eager had the lowest measured warm latency on one and four B200s.",
          },
          {
            id: "eager",
            label: "Eager",
            description:
              "Shorter startup with no graph capture or compilation.",
          },
          {
            id: "compile",
            label: "Compile",
            description:
              "Compile at startup; generated details can vary across fresh processes.",
            disabled: (s) => s.mode !== "t2va",
            disableReason:
              "This picker provides compiled warmup recipes for text generation.",
          },
          {
            id: "bcg",
            label: "CUDA graphs",
            description:
              "Replay captured shapes; other signatures use eager execution.",
            disabled: (s) => s.mode !== "t2va",
            disableReason:
              "Conditioned graph capture requires task-specific warmup inputs; use the eager recipe here.",
          },
        ],
      },
      {
        id: "mode",
        title: "Task",
        scope: "base",
        default: "t2va",
        description:
          "Reference tasks use the Ref2VA adapter. Changing partition requires a server restart.",
        learnMore:
          "/cookbook/diffusion/MiniMax/Pika-H3-Flex#2-model-capabilities",
        options: [
          { id: "t2va", label: "Text" },
          { id: "i2va", label: "First frame" },
          { id: "l2va", label: "Last frame" },
          { id: "fl2va", label: "Both frames" },
          { id: "ref2va", label: "Reference image" },
          { id: "mixed", label: "Image + video + audio" },
        ],
      },
      {
        id: "resolution",
        title: "Resolution tier",
        scope: "request",
        default: "480",
        description: "Use one of the two released canvas tiers.",
        options: [
          { id: "480", label: "480p" },
          { id: "768", label: "768p" },
        ],
      },
      {
        id: "duration",
        title: "Duration",
        scope: "request",
        default: "5",
        description:
          "24 fps; five seconds resolves to 124 frames and fifteen to 362.",
        options: [
          { id: "5", label: "5 seconds" },
          { id: "15", label: "15 seconds" },
        ],
      },
      {
        id: "aspect",
        title: "Aspect ratio",
        scope: "request",
        default: "16:9",
        description:
          "Keyframes are center-cropped to the selected fixed canvas.",
        options: [
          "1:1",
          "4:3",
          "3:4",
          "16:9",
          "9:16",
          "3:2",
          "2:3",
          "21:9",
          "9:21",
          "5:4",
          "4:5",
        ].map((id) => ({ id, label: id })),
      },
    ],
    commandBuilder: {
      defaultSelection: defaults,
      resource: {
        limits: {
          nodes: { min: 1, max: 1 },
          gpus_per_node: { min: 1, max: 4, allowedValues: [1, 2, 4] },
        },
        verifiedRecipes: [1, 2, 4].map((count) => ({
          id: `b200-${count}`,
          hw: "b200",
          nodes: 1,
          gpus_per_node: count,
          tp_size: 1,
          ulysses_degree: count,
          ring_degree: 1,
          default: count === 1,
        })),
        autoTopology: topologyFor,
        validateTopology: (s, t) => {
          const errors = [];
          if (Number(s.nodes) !== 1)
            errors.push("These recipes cover one node.");
          if (![1, 2, 4].includes(Number(s.gpus_per_node)))
            errors.push("Choose 1, 2, or 4 GPUs.");
          if (
            Number(t.tp_size) *
              Number(t.ulysses_degree) *
              Number(t.ring_degree) !==
            Number(s.gpus_per_node)
          )
            errors.push("TP × Ulysses × Ring must equal the GPU count.");
          if (
            ![1, 2, 4].includes(Number(t.tp_size)) ||
            ![1, 2, 4].includes(Number(t.ulysses_degree)) ||
            Number(t.ring_degree) !== 1
          )
            errors.push(
              "Use TP and Ulysses degrees of 1, 2, or 4, with Ring 1.",
            );
          return errors;
        },
      },
      resolveDeployment: (s) => {
        const t =
          s.topology_mode === "manual"
            ? {
                tp_size: Number(s.tp_size),
                ulysses_degree: Number(s.ulysses_degree),
                ring_degree: Number(s.ring_degree),
              }
            : topologyFor(s);
        const errors = config.commandBuilder.resource.validateTopology(s, t);
        const reference = ["ref2va", "mixed"].includes(s.mode);
        const count = Number(s.gpus_per_node);
        const measuredText =
          s.hw === "b200" &&
          s.mode === "t2va" &&
          s.resolution === "480" &&
          s.duration === "5" &&
          s.aspect === "16:9" &&
          [1, 4].includes(count) &&
          (t.tp_size === 1 || (count === 4 && t.tp_size === 2)) &&
          t.ulysses_degree === count / t.tp_size;
        const execution = s.execution === "auto" ? "eager" : s.execution;
        if (s.mode !== "t2va" && execution !== "eager")
          errors.push(
            "Choose Automatic or Eager for this task's startup recipe.",
          );
        const flags = [
          "--model-path Pika-Labs/Pika-H3-Flex",
          `--model-variant ${reference ? "ref2va" : "fl2va"}`,
          "--attention-backend fa",
          `--enable-torch-compile ${execution === "compile"}`,
          `--enable-breakable-cuda-graph ${execution === "bcg"}`,
          `--warmup-mode ${execution === "eager" ? "off" : "server"}`,
          "--host 0.0.0.0",
          "--port {{PORT}}",
        ];
        if (execution !== "eager")
          flags.push(
            "--warmup-resolutions 832x480",
            "--warmup-num-frames 124",
            "--warmup-steps 6",
            `--warmup-sampling-params '${JSON.stringify({ prompt })}'`,
          );
        if (execution === "bcg") flags.push("--bcg-text-buckets 128");
        if (count > 1)
          flags.push(
            `--num-gpus ${count}`,
            `--tp-size ${t.tp_size}`,
            `--ulysses-degree ${t.ulysses_degree}`,
          );
        if (s.hw === "h100" && count === 1)
          flags.push(
            "--performance-mode memory",
            "--layerwise-offload-components dit,text_encoder,vae",
          );
        else flags.push("--performance-mode speed");
        const topologyVerified =
          s.hw === "b200" &&
          Number(s.nodes) === 1 &&
          (t.tp_size === 1 || (count === 4 && t.tp_size === 2)) &&
          t.ulysses_degree === count / t.tp_size;
        const shapeVerified =
          s.aspect === "16:9" &&
          ((s.resolution === "480" && s.duration === "5") ||
            (count === 1 &&
              s.mode === "t2va" &&
              s.resolution === "768" &&
              s.duration === "15"));
        const verified =
          errors.length === 0 &&
          topologyVerified &&
          shapeVerified &&
          (count === 1 || s.mode === "t2va") &&
          (execution === "eager" || measuredText);
        const status = errors.length
          ? "error"
          : verified
            ? "verified"
            : "unverified";
        return {
          match: { hw: s.hw },
          nnodes: 1,
          flags,
          verified,
          builder: {
            topology: t,
            topologySummary: `TP ${t.tp_size} / Ulysses ${t.ulysses_degree} · ${execution}`,
            errors,
            warnings: verified
              ? []
              : [
                  s.hw === "b200"
                    ? "This exact task, canvas, and topology combination has not been measured."
                    : "Hopper source paths have been audited; this Pika recipe has not been run on H100/H200 hardware.",
                ],
            verification: { serve: status, request: status },
          },
        };
      },
    },
    modelNames: { default: "Pika-Labs/Pika-H3-Flex" },
    placeholders: {
      PORT: { target: "command", label: "Server port", default: "30010" },
      CURL_HOST: { target: "curl", label: "Server host", default: "127.0.0.1" },
      CURL_PORT: { target: "curl", label: "Server port", default: "30010" },
      FIRST_FRAME: {
        target: "curl",
        label: "First frame or reference image URI",
        default: "file:///data/first.png",
      },
      LAST_FRAME: {
        target: "curl",
        label: "Last frame URI",
        default: "file:///data/last.png",
      },
      REFERENCE_VIDEO: {
        target: "curl",
        label: "2–15 second reference video URI",
        default: "file:///data/reference.mp4",
      },
      REFERENCE_AUDIO: {
        target: "curl",
        label: "2–15 second reference audio URI",
        default: "file:///data/reference.wav",
      },
    },
    curl: (s) => {
      const task = s.mode === "mixed" ? "ref2va" : s.mode;
      const conditions = [];
      if (["i2va", "fl2va"].includes(task))
        conditions.push({
          type: "image",
          uri: "{{FIRST_FRAME}}",
          role: "keyframe",
          frame_index: 0,
        });
      if (["l2va", "fl2va"].includes(task))
        conditions.push({
          type: "image",
          uri: "{{LAST_FRAME}}",
          role: "keyframe",
          frame_index: -1,
        });
      if (task === "ref2va")
        conditions.push({
          type: "image",
          uri: "{{FIRST_FRAME}}",
          role: "reference",
        });
      if (s.mode === "mixed")
        conditions.push(
          { type: "video", uri: "{{REFERENCE_VIDEO}}", role: "reference" },
          { type: "audio", uri: "{{REFERENCE_AUDIO}}", role: "reference" },
        );
      const body = {
        model: "Pika-Labs/Pika-H3-Flex",
        prompt,
        task,
        conditions,
        target: {
          short_edge: Number(s.resolution),
          aspect_ratio: s.aspect,
          duration_seconds: Number(s.duration),
        },
        num_inference_steps: 6,
        quality: "lossless",
        seed: 42,
      };
      return `curl -sS http://{{CURL_HOST}}:{{CURL_PORT}}/v1/videos \\\n  -H 'Content-Type: application/json' \\\n  -d '${JSON.stringify(body, null, 2)}'`;
    },
    runModes: () => ["python"],
    showPlaygroundLink: false,
    cells: [],
  };
  return config;
})();
