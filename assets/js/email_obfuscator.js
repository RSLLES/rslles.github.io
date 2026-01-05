'use strict';

document.addEventListener("DOMContentLoaded", function () {
    const emails = document.querySelectorAll('.social-icon-Email');

    function text_conversion(input_text) {
        return input_text
            .replace('Lo', 'con')
            .replace('ad', 'tact@')
            .replace('in', 'romai')
            .replace('g.', 'nseail')
            .replace('..', 'les.fr');
    };

    emails.forEach(function (email) {
        // Text
        const text = email.querySelector('.social-text');
        if (text) {
            text.textContent = text_conversion(text.textContent);
        }

        // Href
        email.setAttribute('href', text_conversion(email.getAttribute('href')));
    });
});