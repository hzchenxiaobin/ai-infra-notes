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
});
