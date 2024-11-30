const express = require('express');
const app = express();

//펌웨어 다운로드
app.get('/uploads/*', function (req, res) {
    var uri = url.parse(req.url, true);
    var pathname = uri.pathname;
    var fileurl = pathname.replace('/uploads/', '');
    const file = `${__dirname}/uploads/` + fileurl;

    console.log('firwmware download : ', uri.pathname);

    res.download(file);
});