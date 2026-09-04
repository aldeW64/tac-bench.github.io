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

const previewVideos = document.querySelectorAll("video:not([controls])");

previewVideos.forEach((video) => {
  video.addEventListener("mouseenter", () => video.play());
  video.addEventListener("focus", () => video.play());
});

const heroVideo = document.getElementById("hero-project-video");
const heroVideoMask = document.getElementById("hero-video-mask");

if (heroVideo && heroVideoMask) {
  const toggleHeroVideoMask = () => {
    heroVideoMask.classList.toggle("active", heroVideo.currentTime < 16);
  };

  ["loadedmetadata", "timeupdate", "seeked", "play", "pause", "ended"].forEach((eventName) => {
    heroVideo.addEventListener(eventName, toggleHeroVideoMask);
  });

  toggleHeroVideoMask();
}
