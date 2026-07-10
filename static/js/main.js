const tabs = document.querySelectorAll(".tab[data-filter]");
const taskCards = document.querySelectorAll(".task-grid .video-card");

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const filter = tab.dataset.filter;

    tabs.forEach((item) => item.classList.toggle("active", item === tab));
    taskCards.forEach((card) => {
      const visible = filter === "all" || card.dataset.category === filter;
      card.hidden = !visible;
    });
  });
});

const previewVideos = document.querySelectorAll("video:not([controls])");

previewVideos.forEach((video) => {
  video.addEventListener("mouseenter", () => video.play());
  video.addEventListener("focus", () => video.play());
});
