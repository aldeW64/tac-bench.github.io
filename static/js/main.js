const tabs = document.querySelectorAll(".tab[data-filter]");
const taskCards = document.querySelectorAll(".task-grid .video-card");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const filter = tab.dataset.filter;

    tabs.forEach((item) => {
      const selected = item === tab;
      item.classList.toggle("active", selected);
      item.setAttribute("aria-selected", String(selected));
    });
    taskCards.forEach((card) => {
      const visible = filter === "all" || card.dataset.category === filter;
      card.hidden = !visible;
    });
  });
});

const lazyTaskVideos = document.querySelectorAll(".task-card video[data-video-src]");

function loadTaskVideo(video) {
  if (!video.dataset.videoSrc) return;
  const source = document.createElement("source");
  source.src = video.dataset.videoSrc;
  source.type = "video/mp4";
  video.append(source);
  delete video.dataset.videoSrc;
  video.load();
}

if ("IntersectionObserver" in window) {
  const videoObserver = new IntersectionObserver((entries, observer) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      loadTaskVideo(entry.target);
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "400px 0px" });
  lazyTaskVideos.forEach((video) => videoObserver.observe(video));
} else {
  lazyTaskVideos.forEach(loadTaskVideo);
}

lazyTaskVideos.forEach((video) => {
  ["pointerdown", "focus", "mouseenter"].forEach((eventName) => {
    video.addEventListener(eventName, () => loadTaskVideo(video), { once: true });
  });
});

const provenanceCards = document.querySelectorAll(".task-card[data-task-id]");

fetch("source/tasks/manifest.json")
  .then((response) => {
    if (!response.ok) throw new Error(`manifest request failed: ${response.status}`);
    return response.json();
  })
  .then((manifest) => {
    const entries = new Map(manifest.entries.map((entry) => [entry.website_task, entry]));
    provenanceCards.forEach((card) => {
      const entry = entries.get(card.dataset.taskId);
      const output = card.querySelector(".task-provenance p");
      if (!entry || !output) return;
      const evidence = entry.capture.success_evidence === "success-only-store"
        ? "Success-only source trajectory."
        : "Simulator replay success verified.";
      output.textContent = `${entry.store_task}.zarr · seed ${entry.capture.seed} · `
        + `success step ${entry.capture.success_step} · ${entry.layout} · `
        + `${entry.width}×${entry.height} H.264, ${entry.fps} fps · ${evidence}`;
    });
  })
  .catch(() => {
    provenanceCards.forEach((card) => {
      const output = card.querySelector(".task-provenance p");
      if (output) output.textContent = "Capture record unavailable in this preview.";
    });
  });

const previewVideos = document.querySelectorAll("video:not([controls])");

previewVideos.forEach((video) => {
  video.addEventListener("mouseenter", () => video.play());
  video.addEventListener("focus", () => video.play());
});
