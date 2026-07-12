document.addEventListener('DOMContentLoaded', function() {
    // Mobile menu toggle
    const menuToggle = document.querySelector('.menu-toggle');
    const sidebar = document.querySelector('.sidebar');

    if (menuToggle && sidebar) {
        menuToggle.addEventListener('click', function() {
            sidebar.classList.toggle('open');
        });

        // Close sidebar when clicking on a link (mobile)
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    sidebar.classList.remove('open');
                }
            });
        });
    }

    // Scroll sidebar independently on wheel when it has overflow
    if (sidebar) {
        sidebar.addEventListener('wheel', function(e) {
            if (e.deltaY === 0) return;
            const canScrollUp = sidebar.scrollTop > 0;
            const canScrollDown = sidebar.scrollTop + sidebar.clientHeight < sidebar.scrollHeight;
            if ((e.deltaY < 0 && canScrollUp) || (e.deltaY > 0 && canScrollDown)) {
                e.preventDefault();
                sidebar.scrollTop += e.deltaY;
            }
        }, { passive: false });
    }

    // Accordion navigation in sidebar
    function toggleAccordionItem(item) {
        if (!item) return;
        const willExpand = !item.classList.contains('is-expanded');
        item.classList.toggle('is-expanded');
        const isExpanded = item.classList.contains('is-expanded');
        const button = item.querySelector('.nav-accordion-toggle');
        if (button) {
            button.setAttribute('aria-expanded', isExpanded ? 'true' : 'false');
        }
        // When manually expanding a level, collapse its descendants so that
        // each level requires its own click to expand.
        if (willExpand) {
            item.querySelectorAll('.nav-accordion-item.is-expanded').forEach(child => {
                child.classList.remove('is-expanded');
                const childButton = child.querySelector('.nav-accordion-toggle');
                if (childButton) {
                    childButton.setAttribute('aria-expanded', 'false');
                }
            });
        }
    }

    document.querySelectorAll('.nav-accordion-toggle').forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleAccordionItem(button.closest('.nav-accordion-item'));
        });
    });

    // Clicking a week header toggles the accordion; week links don't navigate.
    document.querySelectorAll('.nav-accordion-header').forEach(header => {
        header.addEventListener('click', function(e) {
            const link = e.target.closest('.week-link');
            if (link) {
                e.preventDefault();
                e.stopPropagation();
            }
            // Don't toggle if the click was on a day-link inside the header (future-proofing)
            if (e.target.closest('.day-link')) {
                return;
            }
            toggleAccordionItem(header.closest('.nav-accordion-item'));
            // Prevent nested accordion clicks from toggling parent accordions
            e.stopPropagation();
        });
    });

    // Add copy buttons to code blocks
    document.querySelectorAll('.content pre').forEach(pre => {
        const wrapper = document.createElement('div');
        wrapper.className = 'code-block-wrapper';
        pre.parentNode.insertBefore(wrapper, pre);
        wrapper.appendChild(pre);

        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', async function() {
            const code = pre.querySelector('code');
            const text = code ? code.textContent : pre.textContent;
            try {
                await navigator.clipboard.writeText(text);
                button.textContent = 'Copied!';
                button.classList.add('copied');
                setTimeout(() => {
                    button.textContent = 'Copy';
                    button.classList.remove('copied');
                }, 2000);
            } catch (err) {
                button.textContent = 'Failed';
                setTimeout(() => {
                    button.textContent = 'Copy';
                }, 2000);
            }
        });
        wrapper.appendChild(button);
    });

    // Back to top button
    const backToTop = document.querySelector('.back-to-top');
    if (backToTop) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 300) {
                backToTop.classList.add('visible');
            } else {
                backToTop.classList.remove('visible');
            }
        });

        backToTop.addEventListener('click', function() {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });
    }

    // Random LeetGPU problem picker on overview page
    const randomPickBtn = document.getElementById('random-pick-btn');
    if (randomPickBtn) {
        randomPickBtn.addEventListener('click', function() {
            try {
                const problems = JSON.parse(randomPickBtn.dataset.problems || '[]');
                if (!problems.length) return;
                const p = problems[Math.floor(Math.random() * problems.length)];
                if (p.slug) {
                    window.location.href = 'https://leetgpu.com/challenges/' + encodeURIComponent(p.slug);
                }
            } catch (e) {
                // Ignore malformed data attribute
            }
        });
    }

    // Image lightbox zoom
    initImageLightbox();

    // Enhance interview Q&A section into styled cards
    enhanceInterviewQA();

    // Open all links in new tab, except sidebar links
    document.querySelectorAll('a').forEach(link => {
        if (link.closest('.sidebar')) {
            return;
        }
        link.setAttribute('target', '_blank');
        link.setAttribute('rel', 'noopener noreferrer');
    });
});

function enhanceInterviewQA() {
    const content = document.querySelector('.content');
    if (!content) return;

    content.querySelectorAll('h4').forEach(heading => {
        const details = heading.nextElementSibling;
        if (!details || details.tagName !== 'DETAILS') return;

        // Avoid re-processing
        if (heading.closest('.qa-card')) return;

        const card = document.createElement('div');
        card.className = 'qa-card';

        heading.classList.add('qa-question');
        details.classList.add('qa-answer');

        const summary = details.querySelector('summary');
        if (summary) {
            summary.classList.add('qa-answer-toggle');
        }

        heading.parentNode.insertBefore(card, heading);
        card.appendChild(heading);
        card.appendChild(details);
    });
}

function initImageLightbox() {
    const content = document.querySelector('.content');
    if (!content) return;

    const images = content.querySelectorAll('img');
    if (images.length === 0) return;

    const lightbox = document.createElement('div');
    lightbox.className = 'image-lightbox';
    lightbox.setAttribute('role', 'dialog');
    lightbox.setAttribute('aria-modal', 'true');
    lightbox.setAttribute('aria-label', 'Image preview');
    lightbox.innerHTML = `
        <button class="lightbox-close" aria-label="Close image preview">&times;</button>
        <img src="" alt="">
        <div class="lightbox-caption"></div>
        <div class="lightbox-zoom-hint">滚轮缩放 / 点击关闭</div>
    `;
    document.body.appendChild(lightbox);

    const lightboxImg = lightbox.querySelector('img');
    const lightboxCaption = lightbox.querySelector('.lightbox-caption');
    const closeButton = lightbox.querySelector('.lightbox-close');
    const zoomHint = lightbox.querySelector('.lightbox-zoom-hint');

    let currentScale = 1;
    const MIN_SCALE = 0.5;
    const MAX_SCALE = 5;
    const ZOOM_STEP = 0.15;

    function updateScale(scale) {
        currentScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
        lightboxImg.style.transform = `scale(${currentScale})`;
    }

    function resetScale() {
        currentScale = 1;
        lightboxImg.style.transform = '';
    }

    function openLightbox(img) {
        resetScale();
        lightboxImg.src = img.src;
        lightboxImg.alt = img.alt || '';
        if (img.alt) {
            lightboxCaption.textContent = img.alt;
            lightboxCaption.style.display = 'block';
        } else {
            lightboxCaption.style.display = 'none';
        }
        lightbox.classList.add('active');
        document.body.style.overflow = 'hidden';
    }

    function closeLightbox() {
        lightbox.classList.remove('active');
        document.body.style.overflow = '';
        resetScale();
    }

    images.forEach(img => {
        img.style.cursor = 'pointer';
        img.addEventListener('click', function(e) {
            e.preventDefault();
            openLightbox(img);
        });
    });

    lightbox.addEventListener('click', function(e) {
        if (e.target === lightbox || e.target === lightboxImg) {
            closeLightbox();
        }
    });

    closeButton.addEventListener('click', closeLightbox);

    // Mouse wheel zoom
    lightbox.addEventListener('wheel', function(e) {
        if (!lightbox.classList.contains('active')) return;
        e.preventDefault();
        const delta = e.deltaY > 0 ? -ZOOM_STEP : ZOOM_STEP;
        updateScale(currentScale + delta);
    }, { passive: false });

    document.addEventListener('keydown', function(e) {
        if (e.key === 'Escape' && lightbox.classList.contains('active')) {
            closeLightbox();
        }
    });
}
