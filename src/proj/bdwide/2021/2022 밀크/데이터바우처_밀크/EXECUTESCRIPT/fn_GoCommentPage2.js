function fn_GoCommentPage2(page) {
	
	if(commentPageProgress) {
		return;
	}
	commentPageProgress = true;
	
	var pFilterVal = $("#commentFilterCol").val(); // 히든값 (필터기준)
	var sortCol = $("#commentSort").val(); // 히든값 (정렬기준)
	var commentItemId = $("#commentItemId").val(); // 히든값 (commentItemId)
	var commentSiteNo = $("#commentSiteNo").val(); // 히든값 (사이트 넘버)
	var commentUitemId = $("#commentUitemId").val(); // 히든값 (all)
	if( commentUitemId === 'all' ){
		commentUitemId = '';
	}
	var recomAttrGrpId = $("#recomAttrGrpId").val(); // 히든값
	var recomAttrId = $("#recomAttrId").val(); // 히든값 ("")
	var commentOreItemId = $("#commentOreItemId").val(); // 히든값 ("")
	var commentOreItemReviewYn = $("#commentOreItemReviewYn").val(); // 히든값("N")

	var checkCnt = 0;

	if( recomAttrGrpId !== '' && recomAttrId !== '' ){
		pFilterVal = '10';
	}
	
	$("#commentPage").val(page);
	var url = "/item/ajaxItemCommentList.ssg";
	var paramData = {itemId:commentItemId, siteNo:commentSiteNo, filterCol:pFilterVal, sortCol:sortCol, uitemId:commentUitemId, recomAttrGrpId:recomAttrGrpId, recomAttrId:recomAttrId, page:page, pageSize:"10", oreItemId:commentOreItemId, oreItemReviewYn:commentOreItemReviewYn};

	$.ajax({
		type: "GET",
		url: url,
		data: paramData,
		dataType: "json",
		success: function (data) {
			
			console.log(data);
			return data;
			
		},
		error: function () {
			alert("에러");
			commentPageProgress = false;
		}
	});
}