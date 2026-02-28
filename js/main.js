/* main.js — Mermaid init + copy buttons + timeline scroll */

/* ── Mermaid initialization ─────────────────────────────── */
document.addEventListener("DOMContentLoaded", function () {
  if (typeof mermaid !== "undefined") {
    mermaid.initialize({
      startOnLoad: true,
      theme: "neutral",
      fontFamily: '"Segoe UI", system-ui, -apple-system, Arial, sans-serif',
      fontSize: 13,
      flowchart: { curve: "basis", htmlLabels: true, padding: 12 },
    });
  }

  /* ── Copy buttons ────────────────────────────────────────── */
  document.querySelectorAll(".code-wrap pre").forEach(function (pre) {
    var wrap = pre.closest(".code-wrap");
    var btn = document.createElement("button");
    btn.className = "copy-btn";
    btn.textContent = "Copy";
    wrap.appendChild(btn);

    btn.addEventListener("click", function () {
      var code = pre.querySelector("code") ? pre.querySelector("code").innerText : pre.innerText;
      navigator.clipboard
        .writeText(code)
        .then(function () {
          btn.textContent = "Copied!";
          btn.classList.add("copied");
          setTimeout(function () {
            btn.textContent = "Copy";
            btn.classList.remove("copied");
          }, 2000);
        })
        .catch(function () {
          btn.textContent = "Error";
        });
    });
  });

  /* ── Active nav highlighting on scroll ───────────────────── */
  var sections = document.querySelectorAll("[data-nav-section]");
  var navLinks = document.querySelectorAll(".nav-inner a[href^='#']");

  if (sections.length && navLinks.length) {
    var observer = new IntersectionObserver(
      function (entries) {
        entries.forEach(function (entry) {
          if (entry.isIntersecting) {
            var id = entry.target.getAttribute("id");
            navLinks.forEach(function (a) {
              a.style.background = "";
              a.style.color = "";
              if (a.getAttribute("href") === "#" + id) {
                a.style.background = "var(--blue-light)";
                a.style.color = "var(--blue)";
              }
            });
          }
        });
      },
      { rootMargin: "-20% 0px -70% 0px" }
    );

    sections.forEach(function (s) { observer.observe(s); });
  }
});
